import threading
import main_test
import loop

thread1 = threading.Thread(target=main.start_audio_and_model)
thread2 = threading.Thread(target=loop.loop)

thread1.start()
thread2.start()
