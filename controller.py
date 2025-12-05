import threading

import main
import loop_normal

thread1 = threading.Thread(target=main.start_audio_and_model)
thread2 = threading.Thread(target=loop_normal.loop)

thread1.start()
thread2.start()

# Keep the main thread alive so worker threads can spawn subthreads safely
thread1.join()
thread2.join()
