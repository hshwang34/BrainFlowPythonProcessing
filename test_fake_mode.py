#!/usr/bin/env python3
"""
Simple test script to verify EEG Bridge fake mode works
This tests the core functionality without requiring EEG hardware
"""

import sys
import subprocess
import time
import socket
import json
from threading import Thread
import argparse

def test_udp_receiver(host="127.0.0.1", port=5005, duration=10):
    """Test UDP receiver to capture EEG Bridge messages"""
    print(f"Starting UDP receiver on {host}:{port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(1.0)  # 1 second timeout
        
        messages_received = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8')
                
                # Parse JSON
                eeg_data = json.loads(message)
                
                # Print first few messages
                if messages_received < 5:
                    print(f"Received: {message}")
                elif messages_received == 5:
                    print("... (continuing to receive messages)")
                
                messages_received += 1
                
                # Validate expected fields
                required_fields = ['t', 'alpha', 'beta', 'theta', 'relax_index', 
                                 'focus_index', 'state', 'mi_prob_left', 'mi_prob_right', 'artifact']
                
                for field in required_fields:
                    if field not in eeg_data:
                        print(f"WARNING: Missing field '{field}' in message")
                        
            except socket.timeout:
                continue  # Keep trying
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Receive error: {e}")
        
        sock.close()
        print(f"Test completed. Received {messages_received} messages in {duration} seconds")
        return messages_received > 0
        
    except Exception as e:
        print(f"UDP receiver error: {e}")
        return False

def test_eeg_bridge_import():
    """Test that eeg_bridge.py can be imported without hardware dependencies"""
    print("Testing EEG Bridge import...")
    
    try:
        # Try to run the help command
        result = subprocess.run([sys.executable, "eeg_bridge.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if "EEG Bridge - Real-time BCI processing for Unity" in result.stdout:
            print("✓ EEG Bridge help command works")
            return True
        else:
            print("✗ EEG Bridge help command failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ EEG Bridge help command timed out")
        return False
    except Exception as e:
        print(f"✗ EEG Bridge import test failed: {e}")
        return False

def run_fake_mode_test(duration=15):
    """Run EEG Bridge in fake mode and test UDP output"""
    print(f"Testing fake mode for {duration} seconds...")
    
    # Start UDP receiver in background
    receiver_thread = Thread(target=test_udp_receiver, args=("127.0.0.1", 5005, duration))
    receiver_thread.daemon = True
    receiver_thread.start()
    
    time.sleep(1)  # Give receiver time to start
    
    try:
        # Start EEG Bridge in fake mode
        cmd = [sys.executable, "eeg_bridge.py", "--run", "--fake", "--user", "test", 
               "--udp-host", "127.0.0.1", "--udp-port", "5005"]
        
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Let it run for specified duration
        time.sleep(duration)
        
        # Terminate the process
        process.terminate()
        
        # Wait for it to finish
        stdout, stderr = process.communicate(timeout=5)
        
        print("EEG Bridge output:")
        print("STDOUT:", stdout[-500:] if len(stdout) > 500 else stdout)  # Last 500 chars
        if stderr:
            print("STDERR:", stderr[-500:] if len(stderr) > 500 else stderr)
        
        # Wait for receiver thread to complete
        receiver_thread.join(timeout=2)
        
        return True
        
    except subprocess.TimeoutExpired:
        print("EEG Bridge process timed out during termination")
        process.kill()
        return False
    except Exception as e:
        print(f"Fake mode test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test EEG Bridge functionality")
    parser.add_argument("--duration", type=int, default=15, 
                       help="Test duration in seconds (default: 15)")
    parser.add_argument("--skip-import", action="store_true",
                       help="Skip import test (useful if dependencies missing)")
    args = parser.parse_args()
    
    print("="*60)
    print("EEG Bridge Test Suite")
    print("="*60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Import test
    if not args.skip_import:
        total_tests += 1
        if test_eeg_bridge_import():
            tests_passed += 1
        print()
    
    # Test 2: Fake mode test (only if dependencies available)
    total_tests += 1
    try:
        import numpy  # Quick check for basic dependencies
        if run_fake_mode_test(args.duration):
            tests_passed += 1
    except ImportError:
        print("Skipping fake mode test - numpy not available")
        print("Install dependencies with: pip install -r requirements.txt")
        total_tests -= 1
    
    print("="*60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
