import threading

import main
import loop_normal

def run_flask():
    """Run the Flask web server."""
    print("Starting local transcriber web app on http://127.0.0.1:5001")
    main.app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

thread1 = threading.Thread(target=main.start_audio_and_model)
thread2 = threading.Thread(target=loop_normal.loop)
thread3 = threading.Thread(target=run_flask)

thread1.start()
thread2.start()
thread3.start()

# Keep the main thread alive so worker threads can spawn subthreads safely
thread1.join()
thread2.join()
thread3.join()
