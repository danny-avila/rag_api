#!/usr/bin/env python3
"""
Automated failover testing script.
Uploads PDFs from ~/uploads_rag/pdfs while toggling NVIDIA service availability.
"""

import os
import time
import random
import subprocess
import requests
import threading
import jwt
from pathlib import Path
from datetime import datetime

# Configuration - check environment variables first
def get_config():
    """Get configuration from environment variables or defaults."""
    # RAG API URL configuration
    rag_host = os.getenv("RAG_HOST", "localhost")
    rag_port = os.getenv("RAG_PORT", "8001")
    rag_api_url = os.getenv("RAG_API_URL", f"http://{rag_host}:{rag_port}")
    
    # PDF directory configuration
    upload_dir = os.getenv("RAG_UPLOAD_DIR", str(Path.home() / "uploads_rag"))
    pdf_dir = Path(upload_dir) / "pdfs"
    
    return rag_api_url, pdf_dir

RAG_API_URL, PDF_DIR = get_config()
TEST_USER_ID = "test-failover-user"

class FailoverTester:
    def __init__(self):
        self.upload_results = []
        self.toggle_log = []
        self.stop_toggling = False
        self.jwt_secret = self._get_jwt_secret()
        self.uploaded_file_ids = []
        
    def _get_jwt_secret(self):
        """Get JWT secret from .env.beta or .env file."""
        # Check .env.beta first, then .env
        for env_filename in [".env.beta", ".env"]:
            env_file = Path(__file__).parent / env_filename
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith('JWT_SECRET='):
                            return line.split('=', 1)[1].strip().strip('"\'')
        return None
    
    def _generate_token(self, user_id="test-failover-user", expire_in_minutes=5):
        """Generate short-lived JWT token like LibreChat does."""
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET not found in .env.beta")
            
        payload = {
            "id": user_id,
            "exp": int(time.time()) + (expire_in_minutes * 60)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def toggle_nvidia_service(self):
        """Toggle NVIDIA service availability using iptables."""
        while not self.stop_toggling:
            try:
                # Block NVIDIA port
                self.log("🚫 Blocking NVIDIA port 8003...")
                subprocess.run([
                    "sudo", "iptables", "-A", "OUTPUT", 
                    "-p", "tcp", "--dport", "8003", "-j", "REJECT"
                ], check=True, capture_output=True)
                self.toggle_log.append(("BLOCK", time.time()))
                
                # Wait 10-20 seconds
                block_duration = random.randint(10, 20)
                self.log(f"   ⏳ Port blocked for {block_duration} seconds")
                time.sleep(block_duration)
                
                if self.stop_toggling:
                    break
                    
                # Unblock NVIDIA port
                self.log("✅ Unblocking NVIDIA port 8003...")
                subprocess.run([
                    "sudo", "iptables", "-D", "OUTPUT", 
                    "-p", "tcp", "--dport", "8003", "-j", "REJECT"
                ], check=True, capture_output=True)
                self.toggle_log.append(("UNBLOCK", time.time()))
                
                # Wait 15-25 seconds
                unblock_duration = random.randint(15, 25)
                self.log(f"   ⏳ Port unblocked for {unblock_duration} seconds")
                time.sleep(unblock_duration)
                
            except subprocess.CalledProcessError as e:
                self.log(f"   ⚠️ iptables command failed: {e}")
                time.sleep(5)
            except Exception as e:
                self.log(f"   ❌ Toggle error: {e}")
                time.sleep(5)
    
    def upload_pdf(self, pdf_path):
        """Upload a single PDF to the RAG API using /embed endpoint with multipart form."""
        file_id = f"test-{pdf_path.stem}-{int(time.time())}"
        
        try:
            start_time = time.time()
            
            self.log(f"📄 Uploading {pdf_path.name} (ID: {file_id[:8]}...)")
            
            # Generate JWT token for authentication
            token = self._generate_token(TEST_USER_ID)
            
            # Use multipart form data like LibreChat does
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f, 'application/pdf')}
                data = {
                    'file_id': file_id,
                    'user_id': TEST_USER_ID
                }
                
                response = requests.post(
                    f"{RAG_API_URL}/embed",
                    files=files,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {token}"
                    },
                    timeout=120  # 2 minute timeout
                )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.log(f"   ✅ SUCCESS: {pdf_path.name} uploaded in {duration:.1f}s")
                result = {
                    'file': pdf_path.name,
                    'file_id': file_id,
                    'status': 'SUCCESS',
                    'duration': duration,
                    'status_code': response.status_code,
                    'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
                }
            else:
                self.log(f"   ❌ FAILED: {pdf_path.name} - HTTP {response.status_code}")
                result = {
                    'file': pdf_path.name,
                    'file_id': file_id,
                    'status': 'FAILED',
                    'duration': duration,
                    'status_code': response.status_code,
                    'error': response.text[:500]
                }
                
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            self.log(f"   ⏰ TIMEOUT: {pdf_path.name} after {duration:.1f}s")
            result = {
                'file': pdf_path.name,
                'file_id': file_id,
                'status': 'TIMEOUT',
                'duration': duration,
                'error': 'Request timeout after 2 minutes'
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"   ❌ ERROR: {pdf_path.name} - {str(e)}")
            result = {
                'file': pdf_path.name,
                'file_id': file_id,
                'status': 'ERROR',
                'duration': duration,
                'error': str(e)
            }
        
        self.upload_results.append(result)
        
        # Track successful uploads for cleanup
        if result['status'] == 'SUCCESS':
            self.uploaded_file_ids.append(result['file_id'])
        
        return result
    
    def cleanup_documents(self):
        """Delete all uploaded documents from the test."""
        if not self.uploaded_file_ids:
            self.log("📋 No documents to clean up")
            return
            
        self.log(f"🧹 Cleaning up {len(self.uploaded_file_ids)} uploaded documents...")
        
        try:
            # Generate token for cleanup
            token = self._generate_token(TEST_USER_ID)
            
            # Delete all uploaded documents
            response = requests.delete(
                f"{RAG_API_URL}/documents",
                json=self.uploaded_file_ids,
                headers={
                    "Content-Type": "application/json", 
                    "Authorization": f"Bearer {token}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.log(f"   ✅ Successfully deleted {len(self.uploaded_file_ids)} documents")
            else:
                self.log(f"   ⚠️ Delete request returned {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            self.log(f"   ❌ Failed to delete documents: {str(e)}")
    
    def cleanup_iptables(self):
        """Ensure NVIDIA port is unblocked."""
        try:
            self.log("🧹 Cleaning up iptables rules...")
            # Try to remove the rule (will fail silently if not present)
            subprocess.run([
                "sudo", "iptables", "-D", "OUTPUT", 
                "-p", "tcp", "--dport", "8003", "-j", "REJECT"
            ], capture_output=True)
            self.log("   ✅ NVIDIA port 8003 is now unblocked")
        except:
            pass
    
    def print_summary(self):
        """Print test results summary."""
        self.log("\n" + "="*60)
        self.log("📊 TEST RESULTS SUMMARY")
        self.log("="*60)
        
        # Upload results
        success_count = sum(1 for r in self.upload_results if r['status'] == 'SUCCESS')
        failed_count = sum(1 for r in self.upload_results if r['status'] == 'FAILED')
        timeout_count = sum(1 for r in self.upload_results if r['status'] == 'TIMEOUT')
        error_count = sum(1 for r in self.upload_results if r['status'] == 'ERROR')
        total_count = len(self.upload_results)
        
        self.log(f"📈 UPLOAD STATISTICS:")
        self.log(f"   Total uploads: {total_count}")
        self.log(f"   ✅ Successful: {success_count} ({success_count/total_count*100:.1f}%)")
        self.log(f"   ❌ Failed: {failed_count} ({failed_count/total_count*100:.1f}%)")
        self.log(f"   ⏰ Timeout: {timeout_count} ({timeout_count/total_count*100:.1f}%)")
        self.log(f"   💥 Error: {error_count} ({error_count/total_count*100:.1f}%)")
        
        # Timing statistics
        successful_uploads = [r for r in self.upload_results if r['status'] == 'SUCCESS']
        if successful_uploads:
            avg_duration = sum(r['duration'] for r in successful_uploads) / len(successful_uploads)
            min_duration = min(r['duration'] for r in successful_uploads)
            max_duration = max(r['duration'] for r in successful_uploads)
            
            self.log(f"\n⏱️ TIMING STATISTICS (successful uploads):")
            self.log(f"   Average duration: {avg_duration:.1f}s")
            self.log(f"   Fastest upload: {min_duration:.1f}s")
            self.log(f"   Slowest upload: {max_duration:.1f}s")
        
        # Toggle statistics
        block_count = sum(1 for action, _ in self.toggle_log if action == 'BLOCK')
        unblock_count = sum(1 for action, _ in self.toggle_log if action == 'UNBLOCK')
        
        self.log(f"\n🔀 FAILOVER STATISTICS:")
        self.log(f"   Port blocks: {block_count}")
        self.log(f"   Port unblocks: {unblock_count}")
        
        # Failed uploads detail
        failed_uploads = [r for r in self.upload_results if r['status'] in ['FAILED', 'TIMEOUT', 'ERROR']]
        if failed_uploads:
            self.log(f"\n❌ FAILED UPLOADS DETAIL:")
            for result in failed_uploads:
                self.log(f"   {result['file']} - {result['status']} ({result['duration']:.1f}s)")
                if 'error' in result:
                    self.log(f"      Error: {result['error'][:100]}...")
    
    def run_test(self):
        """Run the automated failover test."""
        # Check if PDF directory exists
        if not PDF_DIR.exists():
            self.log(f"❌ PDF directory not found: {PDF_DIR}")
            return
        
        # Get list of PDF files
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            self.log(f"❌ No PDF files found in {PDF_DIR}")
            return
        
        self.log(f"🎯 Starting failover test with {len(pdf_files)} PDFs")
        self.log(f"📁 PDF directory: {PDF_DIR}")
        self.log(f"🌐 RAG API URL: {RAG_API_URL}")
        
        # Ensure clean start
        self.cleanup_iptables()
        
        # Start the toggle thread
        toggle_thread = threading.Thread(target=self.toggle_nvidia_service, daemon=True)
        toggle_thread.start()
        
        try:
            # Upload PDFs with random delays
            for i, pdf_path in enumerate(pdf_files, 1):
                self.log(f"\n📋 Progress: {i}/{len(pdf_files)}")
                
                # Upload the PDF
                self.upload_pdf(pdf_path)
                
                # Random delay between uploads (3-8 seconds)
                if i < len(pdf_files):  # Don't delay after last upload
                    delay = random.randint(3, 8)
                    self.log(f"   ⏳ Waiting {delay}s before next upload...")
                    time.sleep(delay)
                
        except KeyboardInterrupt:
            self.log("\n⚠️ Test interrupted by user")
            
        finally:
            # Stop toggling and cleanup
            self.stop_toggling = True
            self.cleanup_iptables()
            
            # Wait a moment for toggle thread to finish
            time.sleep(2)
            
            # Clean up uploaded documents
            self.cleanup_documents()
            
            # Print summary
            self.print_summary()


def main():
    """Main function."""
    print("🚀 RAG API Failover Automation Test")
    print("This script will upload PDFs while toggling NVIDIA service availability")
    print("Press Ctrl+C to stop the test at any time\n")
    
    # Check for sudo access (cache credentials)
    try:
        subprocess.run(["sudo", "-v"], check=True, capture_output=True)
        print("✅ Sudo access confirmed for iptables commands")
    except subprocess.CalledProcessError:
        print("❌ This script requires sudo access for iptables commands")
        print("Please run: sudo -v")
        return
    
    # Run the test
    tester = FailoverTester()
    tester.run_test()


if __name__ == "__main__":
    main()