/*
  Compilation commands:

  For debugging:
  clang -g -O2 -o ppm_to_jpg_threaded_debug ppm_to_jpg_threaded.c \
  -I/opt/homebrew/opt/jpeg-turbo/include -L/opt/homebrew/opt/jpeg-turbo/lib -ljpeg -lpthread

  For release mozjpeg:
  clang -O3 -flto -DNDEBUG -o ppm_to_jpg_threaded_mozjpeg ppm_to_jpg_threaded.c \
  -I/opt/homebrew/opt/mozjpeg/include -L/opt/homebrew/opt/mozjpeg/lib -ljpeg -lpthread

  For release:
  clang -O3 -flto -DNDEBUG -o ppm_to_jpg_threaded_jpegturbo ppm_to_jpg_threaded.c \
  -I/opt/homebrew/opt/jpeg-turbo/include -L/opt/homebrew/opt/jpeg-turbo/lib -ljpeg -lpthread

  clang-format -i ppm_to_jpg_threaded.c

  Notes:

  1) mozjpeg slower because it compresses data
*/

#include <dirent.h>  // For directory scanning
#include <errno.h>   // For error numbers
#include <pthread.h> // For threading
#include <stdbool.h> // For bool type
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>   // For nanosleep (used in retry logic)
#include <unistd.h> // For sysconf

#include <jpeglib.h>

#define MAX_READ_ATTEMPTS 5 // REVISION: Number of times to try reading a file.
#define RETRY_DELAY_MS                                                         \
  100 // REVISION: Delay between read attempts in milliseconds.

// Holds the information for a single file conversion task.
typedef struct {
  char *input_path;
  char *output_path;
  int quality;
} ConversionTask;

// A node in the task queue (a simple linked list).
typedef struct TaskNode {
  ConversionTask *task;
  struct TaskNode *next;
} TaskNode;

// The shared task queue for the producer-consumer model.
typedef struct {
  TaskNode *front;
  TaskNode *rear;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool finished_adding; // Flag to signal when the producer is done.
  int count;
} TaskQueue;

// --- Function Prototypes ---
void initialize_queue(TaskQueue *q);
void enqueue(TaskQueue *q, ConversionTask *task);
ConversionTask *dequeue(TaskQueue *q);
void destroy_queue(TaskQueue *q);
void *worker_thread_function(void *arg);
unsigned char *read_ppm(const char *filename, int *width, int *height);

/**
 * @brief This is the worker function that each thread will execute.
 * It continuously pulls tasks from the shared queue and processes them.
 * REVISION: Now includes a retry mechanism to handle partially written files.
 * @param arg A void pointer to the shared TaskQueue.
 * @return Always returns NULL.
 */
void *worker_thread_function(void *arg) {
  TaskQueue *q = (TaskQueue *)arg;

  while (true) {
    pthread_mutex_lock(&q->mutex);
    while (q->front == NULL && !q->finished_adding) {
      pthread_cond_wait(&q->cond, &q->mutex);
    }

    if (q->front == NULL && q->finished_adding) {
      pthread_mutex_unlock(&q->mutex);
      break;
    }

    ConversionTask *task = dequeue(q);
    pthread_mutex_unlock(&q->mutex);

    if (task) {
      const char *ppm_filename = task->input_path;
      const char *jpeg_filename = task->output_path;
      int quality = task->quality;

      unsigned char *image_buffer = NULL;
      int width = 0, height = 0;

      // --- REVISION: Retry Logic ---
      // Attempt to read the file multiple times to handle race conditions
      // where the file is detected before it's fully written to disk.
      for (int attempt = 1; attempt <= MAX_READ_ATTEMPTS; ++attempt) {
        image_buffer = read_ppm(ppm_filename, &width, &height);
        if (image_buffer) {
          break; // Successfully read the file.
        }

        // If read failed, wait and retry.
        if (attempt < MAX_READ_ATTEMPTS) {
          fprintf(stderr,
                  "Warning: Failed read on '%s' (attempt %d/%d). Retrying in "
                  "%dms...\n",
                  ppm_filename, attempt, MAX_READ_ATTEMPTS, RETRY_DELAY_MS);
          struct timespec ts = {.tv_sec = 0,
                                .tv_nsec = RETRY_DELAY_MS * 1000000L};
          nanosleep(&ts, NULL);
        } else {
          fprintf(
              stderr,
              "Error: Failed to read '%s' after %d attempts. Skipping file.\n",
              ppm_filename, MAX_READ_ATTEMPTS);
        }
      }

      if (image_buffer) {
        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);

        FILE *outfile = fopen(jpeg_filename, "wb");
        if (outfile) {
          jpeg_stdio_dest(&cinfo, outfile);
          cinfo.image_width = width;
          cinfo.image_height = height;
          cinfo.input_components = 3;
          cinfo.in_color_space = JCS_RGB;

          jpeg_set_defaults(&cinfo);
          jpeg_set_quality(&cinfo, quality, TRUE);
          jpeg_start_compress(&cinfo, TRUE);

          JSAMPROW row_pointer[1];
          int row_stride = width * 3;
          while (cinfo.next_scanline < cinfo.image_height) {
            row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
          }

          jpeg_finish_compress(&cinfo);
          fclose(outfile);
          printf("Successfully converted %s -> %s\n", ppm_filename,
                 jpeg_filename);
        } else {
          fprintf(stderr, "Error: Cannot open output file %s\n", jpeg_filename);
        }
        jpeg_destroy_compress(&cinfo);
        free(image_buffer);
      }
      // Clean up the task's memory
      free(task->input_path);
      free(task->output_path);
      free(task);
    }
  }
  return NULL;
}

/**
 * @brief The main entry point of the program (The "Producer").
 */
int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <folder_path> <quality>\n", argv[0]);
    fprintf(stderr, "  folder_path: The directory containing .ppm files.\n");
    fprintf(stderr, "  quality:     JPEG quality from 1 to 100.\n");
    return 1;
  }

  const int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  printf("Using %d worker threads.\n", NUM_THREADS);

  const char *folder_path = argv[1];
  int quality = atoi(argv[2]);

  if (quality < 1 || quality > 100) {
    fprintf(stderr, "Error: Quality must be between 1 and 100.\n");
    return 1;
  }

  TaskQueue queue;
  initialize_queue(&queue);
  pthread_t threads[NUM_THREADS];

  printf("Starting %d worker threads...\n", NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++) {
    if (pthread_create(&threads[i], NULL, worker_thread_function, &queue) !=
        0) {
      fprintf(stderr, "Error: Failed to create worker thread %d\n", i);
      return 1;
    }
  }

  DIR *d = opendir(folder_path);
  if (!d) {
    fprintf(stderr, "Error: Cannot open directory '%s': %s\n", folder_path,
            strerror(errno));
    return 1;
  }

  printf("Scanning directory '%s' for PPM files...\n", folder_path);
  struct dirent *dir;
  while ((dir = readdir(d)) != NULL) {
    const char *ext = strrchr(dir->d_name, '.');
    if (ext && strcmp(ext, ".ppm") == 0) {
      ConversionTask *task = (ConversionTask *)malloc(sizeof(ConversionTask));
      if (!task) {
        fprintf(stderr,
                "Error: Failed to allocate memory for a conversion task.\n");
        continue;
      }

      size_t input_path_len = strlen(folder_path) + 1 + strlen(dir->d_name) + 1;
      task->input_path = (char *)malloc(input_path_len);
      snprintf(task->input_path, input_path_len, "%s/%s", folder_path,
               dir->d_name);

      size_t basename_len = ext - dir->d_name;
      size_t output_path_len = strlen(folder_path) + 1 + basename_len + 5;
      task->output_path = (char *)malloc(output_path_len);
      snprintf(task->output_path, output_path_len, "%s/%.*s.jpg", folder_path,
               (int)basename_len, dir->d_name);

      task->quality = quality;

      enqueue(&queue, task);
    }
  }
  closedir(d);

  pthread_mutex_lock(&queue.mutex);
  queue.finished_adding = true;
  pthread_cond_broadcast(&queue.cond);
  pthread_mutex_unlock(&queue.mutex);

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  printf("\nAll conversions finished.\n");
  destroy_queue(&queue);
  return 0;
}

// --- Queue Management Functions (Unchanged) ---

void initialize_queue(TaskQueue *q) {
  q->front = NULL;
  q->rear = NULL;
  q->count = 0;
  q->finished_adding = false;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->cond, NULL);
}

void enqueue(TaskQueue *q, ConversionTask *task) {
  TaskNode *newNode = (TaskNode *)malloc(sizeof(TaskNode));
  newNode->task = task;
  newNode->next = NULL;

  pthread_mutex_lock(&q->mutex);
  if (q->rear) {
    q->rear->next = newNode;
  } else {
    q->front = newNode;
  }
  q->rear = newNode;
  q->count++;

  pthread_cond_signal(&q->cond);
  pthread_mutex_unlock(&q->mutex);
}

ConversionTask *dequeue(TaskQueue *q) {
  if (q->front == NULL)
    return NULL;
  TaskNode *temp = q->front;
  ConversionTask *task = temp->task;
  q->front = q->front->next;
  if (q->front == NULL)
    q->rear = NULL;
  free(temp);
  q->count--;
  return task;
}

void destroy_queue(TaskQueue *q) {
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->cond);
}

/**
 * @brief Reads a binary P6 PPM file into a memory buffer.
 * REVISION: Parsing logic is now more robust, using fgets to read lines
 * instead of relying on specific whitespace in fscanf.
 * @param filename The path to the PPM file.
 * @param width Pointer to store the image width.
 * @param height Pointer to store the image height.
 * @return A pointer to the allocated image buffer on success, or NULL on
 * failure.
 */
unsigned char *read_ppm(const char *filename, int *width, int *height) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    // This error is now less critical as the caller will retry.
    // fprintf(stderr, "Error: Cannot open PPM file %s\n", filename);
    return NULL;
  }

  char line_buffer[256];

  // 1. Read and validate Magic Number (P6)
  if (!fgets(line_buffer, sizeof(line_buffer), fp) ||
      strncmp(line_buffer, "P6", 2) != 0) {
    fprintf(stderr, "Error: Input file '%s' is not a binary P6 PPM.\n",
            filename);
    fclose(fp);
    return NULL;
  }

  // 2. Read dimensions, skipping comments
  bool dims_found = false;
  while (fgets(line_buffer, sizeof(line_buffer), fp)) {
    if (line_buffer[0] == '#') { // Skip comment lines
      continue;
    }
    if (sscanf(line_buffer, "%d %d", width, height) == 2) {
      dims_found = true;
      break;
    }
  }
  if (!dims_found) {
    fprintf(stderr, "Error: Invalid or missing PPM dimensions in '%s'.\n",
            filename);
    fclose(fp);
    return NULL;
  }

  // 3. Read max color value and validate it's 255
  int max_val = 0;
  if (!fgets(line_buffer, sizeof(line_buffer), fp) ||
      sscanf(line_buffer, "%d", &max_val) != 1) {
    fprintf(stderr, "Error: Invalid or missing max color value in '%s'.\n",
            filename);
    fclose(fp);
    return NULL;
  }
  if (max_val != 255) {
    fprintf(stderr,
            "Error: Max color value in '%s' is %d, not 255. Only 24-bit PPMs "
            "are supported.\n",
            filename, max_val);
    fclose(fp);
    return NULL;
  }

  // 4. Read pixel data
  size_t data_size = (size_t)(*width) * (*height) * 3;
  unsigned char *data = (unsigned char *)malloc(data_size);
  if (!data) {
    fprintf(stderr,
            "Error: Could not allocate memory for pixel data for '%s'.\n",
            filename);
    fclose(fp);
    return NULL;
  }

  if (fread(data, 1, data_size, fp) != data_size) {
    // This is the original error point. The caller will now handle this by
    // retrying. fprintf(stderr, "Error: Could not read pixel data from '%s'.
    // File may be truncated.\n", filename);
    free(data);
    fclose(fp);
    return NULL;
  }

  fclose(fp);
  return data;
}
