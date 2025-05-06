import sys
import time
import subprocess
import signal
import threading
import argparse

# Configure the paths
BACKEND_SCRIPT = "agent_langchain.py"
FRONTEND_SCRIPT = "frontend_interface.py"
BACKEND_PORT = 8008
FRONTEND_PORT = 5005

# Global process handles
backend_process = None
frontend_process = None

def start_backend():
    """Start the backend server"""
    global backend_process
    print(f"Starting backend server on port {BACKEND_PORT}...")
    # Use subprocess to start the backend
    backend_process = subprocess.Popen(
        [sys.executable, BACKEND_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    return backend_process

def start_frontend():
    """Start the frontend server"""
    global frontend_process
    print(f"Starting frontend server on port {FRONTEND_PORT}...")
    # Use subprocess to start the frontend
    frontend_process = subprocess.Popen(
        [sys.executable, FRONTEND_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    return frontend_process

def monitor_process(process, name):
    """Monitor a process and print its output"""
    for line in iter(process.stdout.readline, ''):
        print(f"[{name}] {line.strip()}")
    
    if process.poll() is not None:
        print(f"[{name}] Process terminated with code {process.returncode}")

def cleanup(signum=None, frame=None):
    """Clean up and terminate child processes"""
    print("\nShutting down servers...")
    
    # Terminate processes
    if backend_process:
        backend_process.terminate()
        print("Backend server terminated.")
    
    if frontend_process:
        frontend_process.terminate()
        print("Frontend server terminated.")
    
    # Wait for processes to exit
    if backend_process:
        backend_process.wait()
    
    if frontend_process:
        frontend_process.wait()
    
    print("All servers shut down. Exiting.")
    sys.exit(0)

def main():
    """Main function to start the application"""
    parser = argparse.ArgumentParser(description='Start the Sepsis EHR AI Agent')
    parser.add_argument('--backend-only', action='store_true', help='Start only the backend server')
    parser.add_argument('--frontend-only', action='store_true', help='Start only the frontend server')
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Start servers based on command line arguments
        if args.backend_only:
            # Start only the backend
            backend = start_backend()
            threading.Thread(target=monitor_process, args=(backend, "BACKEND"), daemon=True).start()
        elif args.frontend_only:
            # Start only the frontend
            frontend = start_frontend()
            threading.Thread(target=monitor_process, args=(frontend, "FRONTEND"), daemon=True).start()
        else:
            # Start both servers
            backend = start_backend()
            # Wait a bit for the backend to start up
            time.sleep(2)
            frontend = start_frontend()
            
            # Start monitoring threads
            threading.Thread(target=monitor_process, args=(backend, "BACKEND"), daemon=True).start()
            threading.Thread(target=monitor_process, args=(frontend, "FRONTEND"), daemon=True).start()
        
        print("\nServers are running!")
        print(f"- Backend API server: http://localhost:{BACKEND_PORT}")
        print(f"- Frontend web interface: http://localhost:{FRONTEND_PORT}")
        print("\nPress Ctrl+C to stop all servers.")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
            # Check if the processes are still running
            if backend_process and backend_process.poll() is not None:
                print(f"Backend server stopped unexpectedly with code {backend_process.returncode}")
                if not args.backend_only:
                    cleanup()
                    break
            
            if frontend_process and frontend_process.poll() is not None:
                print(f"Frontend server stopped unexpectedly with code {frontend_process.returncode}")
                if not args.frontend_only:
                    cleanup()
                    break
    
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        cleanup()

if __name__ == "__main__":
    main()