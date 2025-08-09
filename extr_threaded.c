/*
  clang -g -O2 -o frame_extractor_threaded_debug extr_threaded.c $(pkg-config \
  --cflags --libs libavformat libavcodec libswscale libavutil) -lpthread

  clang -O3 -flto -DNDEBUG -o frame_extractor_threaded extr_threaded.c \
  $(pkg-config --cflags --libs libavformat libavcodec libswscale libavutil) \
  -lpthread

  clang-format -i extr_threaded.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // For sysconf

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

#include <pthread.h>

// ==============================================================================
// Task Queue for Threading
// ==============================================================================

// A task for a worker thread to complete. Now it's just saving a raw frame.
typedef struct {
  AVFrame *frame; // The RGB frame to be saved.
  int sec;        // The timestamp (in seconds) for the filename.
} SaveTask;

// A simple, thread-safe queue for SaveTasks.
typedef struct {
  SaveTask **tasks;
  int capacity;
  int size;
  int head;
  int tail;
  pthread_mutex_t mutex;
  pthread_cond_t cond_full;
  pthread_cond_t cond_empty;
  int finished; // Flag to signal that no more tasks will be added.
} TaskQueue;

// Initializes the task queue.
void queue_init(TaskQueue *q, int capacity) {
  q->tasks = malloc(sizeof(SaveTask *) * capacity);
  q->capacity = capacity;
  q->size = 0;
  q->head = 0;
  q->tail = 0;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->cond_full, NULL);
  pthread_cond_init(&q->cond_empty, NULL);
  q->finished = 0;
}

// Pushes a task to the queue. Blocks if the queue is full.
void queue_push(TaskQueue *q, SaveTask *task) {
  pthread_mutex_lock(&q->mutex);
  while (q->size == q->capacity) {
    pthread_cond_wait(&q->cond_full, &q->mutex);
  }
  q->tasks[q->tail] = task;
  q->tail = (q->tail + 1) % q->capacity;
  q->size++;
  pthread_cond_signal(&q->cond_empty);
  pthread_mutex_unlock(&q->mutex);
}

// Pops a task from the queue. Blocks if the queue is empty.
// Returns NULL if the queue is empty and production is finished.
SaveTask *queue_pop(TaskQueue *q) {
  pthread_mutex_lock(&q->mutex);
  while (q->size == 0) {
    if (q->finished) {
      pthread_mutex_unlock(&q->mutex);
      return NULL; // No more tasks will be added.
    }
    pthread_cond_wait(&q->cond_empty, &q->mutex);
  }
  SaveTask *task = q->tasks[q->head];
  q->head = (q->head + 1) % q->capacity;
  q->size--;
  pthread_cond_signal(&q->cond_full);
  pthread_mutex_unlock(&q->mutex);
  return task;
}

// Signals that no more tasks will be added and wakes up all waiting threads.
void queue_finish(TaskQueue *q) {
  pthread_mutex_lock(&q->mutex);
  q->finished = 1;
  pthread_cond_broadcast(&q->cond_empty); // Wake up all waiting workers.
  pthread_mutex_unlock(&q->mutex);
}

// Destroys the queue and its resources.
void queue_destroy(TaskQueue *q) {
  // Free any remaining tasks in the queue
  // This is important if the producer thread fails mid-way
  if (q && q->tasks) {
    while (q->size > 0) {
      SaveTask *task = q->tasks[q->head];
      if (task) {
        av_frame_free(&task->frame);
        free(task);
      }
      q->head = (q->head + 1) % q->capacity;
      q->size--;
    }
    free(q->tasks);
    q->tasks = NULL;
  }
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->cond_full);
  pthread_cond_destroy(&q->cond_empty);
}

// ==============================================================================
// Frame Saving and Worker Logic
// ==============================================================================

// Helper function to save a raw RGB AVFrame to a .ppm file.
void save_frame_as_ppm(AVFrame *pFrame, int sec) {
  char szFilename[256];
  snprintf(szFilename, sizeof(szFilename), "frame_at_%ds.ppm", sec);
  FILE *pFile = fopen(szFilename, "wb");
  if (pFile == NULL) {
    fprintf(stderr, "Error: Could not open %s\n", szFilename);
    return;
  }

  // Write the PPM header
  fprintf(pFile, "P6\n%d %d\n255\n", pFrame->width, pFrame->height);

  // Write the pixel data row by row to account for linesize padding
  for (int y = 0; y < pFrame->height; y++) {
    fwrite(pFrame->data[0] + y * pFrame->linesize[0], 1, pFrame->width * 3,
           pFile);
  }

  fclose(pFile);
  printf("Saved frame as %s\n", szFilename);
}

// Data passed to each worker thread.
typedef struct {
  int thread_id;
  TaskQueue *queue;
} WorkerData;

// The function executed by each worker thread.
void *worker_function(void *arg) {
  WorkerData *data = (WorkerData *)arg;
  TaskQueue *queue = data->queue;

  while (1) {
    SaveTask *task = queue_pop(queue);
    if (task == NULL) {
      // This means the queue is empty and production is finished.
      break;
    }

    // Process the task: save the frame as PPM
    save_frame_as_ppm(task->frame, task->sec);

    // Cleanup the task resources
    av_frame_free(&task->frame);
    free(task);
  }

  printf("Thread %d finished.\n", data->thread_id);
  return NULL;
}

// ==============================================================================
// Main Application Logic
// ==============================================================================

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <input_video_file>\n", argv[0]);
    return -1;
  }

  // --- Variable Initialization ---
  int ret = -1; // Default return code to error
  const int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  const char *filename = argv[1];

  // Initialize all resource pointers to NULL.
  // This allows the cleanup section to safely call free functions
  // even if the allocation failed.
  AVFormatContext *pFormatCtx = NULL;
  AVCodecContext *pDecoderCtx = NULL;
  const AVCodec *pDecoder = NULL;
  AVFrame *pFrame = NULL;
  AVFrame *pFrameRGB = NULL;
  AVPacket *pPacket = NULL;
  struct SwsContext *sws_ctx = NULL;
  uint8_t *buffer = NULL;
  TaskQueue queue = {0}; // Zero-initialize the queue struct
  pthread_t *workers = NULL;
  WorkerData *worker_data = NULL;

  printf("Using %d worker threads.\n", NUM_THREADS);

  // 1. Open video file and retrieve stream information
  if (avformat_open_input(&pFormatCtx, filename, NULL, NULL) != 0) {
    fprintf(stderr, "Error: Could not open file %s\n", filename);
    goto cleanup; // Use goto for centralized cleanup
  }
  if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {
    fprintf(stderr, "Error: Could not find stream information\n");
    goto cleanup;
  }
  av_dump_format(pFormatCtx, 0, filename, 0);

  // 2. Find the first video stream
  int videoStreamIndex = -1;
  AVStream *videoStream = NULL;
  for (int i = 0; i < pFormatCtx->nb_streams; i++) {
    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStreamIndex = i;
      videoStream = pFormatCtx->streams[i];
      break;
    }
  }
  if (videoStreamIndex == -1) {
    fprintf(stderr, "Error: Did not find a video stream\n");
    goto cleanup;
  }

  // 3. Get video properties: duration
  int duration_secs = 0;
  if (pFormatCtx->duration != AV_NOPTS_VALUE) {
    duration_secs = (pFormatCtx->duration / AV_TIME_BASE);
    printf("Total duration: %d seconds\n", duration_secs);
  } else {
    printf("Total duration: N/A\n");
    goto cleanup;
  }

  // 4. Setup the DECODER
  pDecoder = avcodec_find_decoder(videoStream->codecpar->codec_id);
  if (!pDecoder) {
    fprintf(stderr, "Error: Unsupported codec!\n");
    goto cleanup;
  }
  pDecoderCtx = avcodec_alloc_context3(pDecoder);
  if (!pDecoderCtx) {
    fprintf(stderr, "Error: Could not allocate decoder context\n");
    goto cleanup;
  }
  avcodec_parameters_to_context(pDecoderCtx, videoStream->codecpar);
  if (avcodec_open2(pDecoderCtx, pDecoder, NULL) < 0) {
    fprintf(stderr, "Error: Could not open decoder\n");
    goto cleanup; // This was a major leak point in the original code
  }

  // 5. Initialize Task Queue and Worker Threads
  queue_init(&queue, NUM_THREADS * 2);
  workers = malloc(sizeof(pthread_t) * NUM_THREADS);
  worker_data = malloc(sizeof(WorkerData) * NUM_THREADS);
  if (!workers || !worker_data) {
    fprintf(stderr, "Error: Could not allocate memory for threads\n");
    goto cleanup;
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    worker_data[i].thread_id = i;
    worker_data[i].queue = &queue;
    if (pthread_create(&workers[i], NULL, worker_function, &worker_data[i]) !=
        0) {
      fprintf(stderr, "Error: Failed to create thread %d\n", i);
      // In a real-world scenario, you might want to handle this more
      // gracefully, but for this example, we'll just clean up and exit. Note:
      // Not all threads may have been created, so we can't join them all. The
      // simplest solution here is to exit. The OS will clean up created
      // threads.
      goto cleanup;
    }
  }

  // 6. Allocate necessary frames and buffers for the producer
  pFrame = av_frame_alloc();
  pFrameRGB = av_frame_alloc();
  pPacket = av_packet_alloc();
  if (!pFrame || !pFrameRGB || !pPacket) {
    fprintf(stderr, "Error: Could not allocate frames or packet\n");
    goto cleanup;
  }

  // Allocate buffer for the RGB frame
  int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pDecoderCtx->width,
                                          pDecoderCtx->height, 32);
  buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  if (!buffer) {
    fprintf(stderr, "Error: Could not allocate RGB buffer\n");
    goto cleanup;
  }
  av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer,
                       AV_PIX_FMT_RGB24, pDecoderCtx->width,
                       pDecoderCtx->height, 32);
  pFrameRGB->width = pDecoderCtx->width;
  pFrameRGB->height = pDecoderCtx->height;
  pFrameRGB->format = AV_PIX_FMT_RGB24;

  // Initialize SWS context for converting to RGB24
  sws_ctx = sws_getContext(pDecoderCtx->width, pDecoderCtx->height,
                           pDecoderCtx->pix_fmt, pDecoderCtx->width,
                           pDecoderCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                           NULL, NULL, NULL);
  if (!sws_ctx) {
    fprintf(stderr, "Error: Could not initialize SWS context\n");
    goto cleanup;
  }

  // 7. Loop through video, decode frames, and push tasks to the queue
  printf("Starting frame extraction...\n");
  for (int i = 0; i < duration_secs; i++) {
    int64_t seek_target = (int64_t)i * AV_TIME_BASE;

    if (av_seek_frame(pFormatCtx, videoStreamIndex, seek_target,
                      AVSEEK_FLAG_BACKWARD) < 0) {
      fprintf(stderr, "Warning: Error seeking to second %d\n", i);
      continue;
    }
    avcodec_flush_buffers(pDecoderCtx);

    while (av_read_frame(pFormatCtx, pPacket) >= 0) {
      if (pPacket->stream_index == videoStreamIndex) {
        if (avcodec_send_packet(pDecoderCtx, pPacket) != 0)
          break;

        int receive_ret = avcodec_receive_frame(pDecoderCtx, pFrame);
        if (receive_ret == 0) {
          // Convert the frame to RGB format for PPM
          sws_scale(sws_ctx, (uint8_t const *const *)pFrame->data,
                    pFrame->linesize, 0, pDecoderCtx->height, pFrameRGB->data,
                    pFrameRGB->linesize);

          // Create a task for the worker
          SaveTask *task = malloc(sizeof(SaveTask));
          if (!task) {
            fprintf(stderr, "Error: Failed to allocate SaveTask\n");
            // This is a critical error, we should stop processing.
            goto producer_done;
          }
          task->sec = i;
          task->frame = av_frame_clone(pFrameRGB);
          if (!task->frame) {
            fprintf(stderr, "Error: Failed to clone AVFrame\n");
            free(task); // Free the task struct itself
            goto producer_done;
          }

          // Push the task to the queue
          queue_push(&queue, task);

          break; // Found the frame for this second, move to the next second
        } else if (receive_ret == AVERROR(EAGAIN) ||
                   receive_ret == AVERROR_EOF) {
          continue; // Need more packets to get a frame
        } else {
          fprintf(stderr, "Warning: Error receiving frame at second %d\n", i);
          break; // Error, move to the next second
        }
      }
      av_packet_unref(pPacket);
    }
    av_packet_unref(pPacket); // Unref packet if loop exits without unref-ing
  }

producer_done:
  // 8. Signal workers that we're done and wait for them to finish
  printf("Finished producing tasks. Waiting for workers to complete...\n");
  queue_finish(&queue);
  if (workers) {
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(workers[i], NULL);
    }
  }

  ret = 0; // Success

cleanup:
  // 9. Cleanup all allocated resources
  printf("Cleaning up resources.\n");

  // The queue needs to be destroyed after threads are joined
  // to avoid race conditions.
  queue_destroy(&queue);

  free(workers);
  free(worker_data);

  av_free(buffer); // Corresponds to av_malloc
  av_frame_free(&pFrameRGB);
  av_frame_free(&pFrame);
  av_packet_free(&pPacket);
  sws_freeContext(sws_ctx);
  avcodec_free_context(&pDecoderCtx);
  avformat_close_input(&pFormatCtx); // This also frees pFormatCtx

  printf(ret == 0 ? "Cleanup complete. Exiting successfully.\n"
                  : "Cleanup complete. Exiting with error.\n");
  return ret;
}
