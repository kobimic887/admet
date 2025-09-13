import os
import sys
import ssl
import json
import logging
import requests
from urllib.parse import urlparse

import pika

# QUIET LOGGING (only errors)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Import ADMET model (suppress possible model init chatter if needed)
from admet_ai import ADMETModel

class AMQPAdmetReceiver:
    def __init__(self, amqp_url: str, queue_name: str, batch_mode: bool = True, api_base_url: str = None):
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self.batch_mode = batch_mode
        self.model = None
        self.api_base_url = api_base_url or "https://app.pyxis-discovery.com"

    def connect(self):
        url = urlparse(self.amqp_url)
        credentials = pika.PlainCredentials(url.username, url.password)

        ssl_options = None
        if url.scheme == 'amqps':
            context = ssl.create_default_context()
            # NOTE: For production you should validate certificates.
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            ssl_options = pika.SSLOptions(context)

        params = pika.ConnectionParameters(
            host=url.hostname,
            port=url.port or (5671 if url.scheme == 'amqps' else 5672),
            virtual_host=url.path[1:] if url.path else '/',
            credentials=credentials,
            ssl_options=ssl_options
        )

        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)

        # Load model once
        self.model = ADMETModel()

    def parse_message_payload(self, raw: str):
        """
        Parse incoming message to extract simulation_id and SMILES data.
        Expected format: {"simulation_id": "abc123def456", "smiles": ["CCO", "CCN"]}
        Or: {"simulation_id": "abc123def456", "smiles": "CCO"}
        Returns: (simulation_id, smiles_list)
        """
        raw = raw.strip()
        if not raw:
            return None, []

        # Try to clean up common malformed patterns
        if raw.startswith('{{') or raw.startswith('{'):
            # Try to extract simulation_id and smiles from malformed data
            import re
            
            # Extract simulation_id
            sim_match = re.search(r'simulationId[:\s]*([a-zA-Z0-9]+)', raw)
            simulation_id = sim_match.group(1) if sim_match else "unknown"
            
            # Extract SMILES
            smiles_match = re.search(r'smiles[:\s]*([A-Za-z0-9()=\[\]@+\-#]+)', raw)
            if smiles_match:
                smiles_list = [smiles_match.group(1)]
                return simulation_id, smiles_list

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                simulation_id = data.get('simulation_id')
                smiles_data = data.get('smiles', [])
                
                if not simulation_id:
                    logger.error("No simulation_id found in message")
                    return None, []
                
                # Parse SMILES data
                if isinstance(smiles_data, list):
                    smiles_list = [s.strip() for s in smiles_data if s.strip()]
                elif isinstance(smiles_data, str):
                    smiles_list = [smiles_data.strip()] if smiles_data.strip() else []
                else:
                    smiles_list = []
                
                return simulation_id, smiles_list
        except json.JSONDecodeError:
            # Fall back to old format - treat as raw SMILES
            smiles_list = self.parse_smiles_payload(raw)
            return "unknown", smiles_list
        
        return None, []

    def parse_smiles_payload(self, raw: str):
        """
        Accepts:
          - Single SMILES string
          - Newline-separated SMILES
          - Comma-separated SMILES
          - JSON list: ["CCO","CCN"]
          - JSON object with key 'smiles' as list or string
        Returns list[str]
        """
        raw = raw.strip()
        if not raw:
            return []

        # Try JSON first
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [s.strip() for s in data if s.strip()]
            if isinstance(data, dict):
                if 'smiles' in data:
                    val = data['smiles']
                    if isinstance(val, list):
                        return [s.strip() for s in val if s.strip()]
                    if isinstance(val, str):
                        return [val.strip()] if val.strip() else []
            # Fallback if JSON parsed but not recognized
        except json.JSONDecodeError:
            pass

        # Try newline separated
        if '\n' in raw:
            parts = [p.strip() for p in raw.splitlines() if p.strip()]
            if len(parts) > 1:
                return parts

        # Try comma separated
        if ',' in raw:
            parts = [p.strip() for p in raw.split(',') if p.strip()]
            if len(parts) > 1:
                return parts

        # Single SMILES
        return [raw]

    def send_predictions_to_api(self, simulation_id: str, predictions):
        """
        Send ADMET predictions to the API endpoint.
        predictions: raw output from model.predict()
        """
        # Try different URL formats
        urls_to_try = [
            f"{self.api_base_url}/api/simulation/{simulation_id}/admet",
            f"{self.api_base_url}:3000/api/simulation/{simulation_id}/admet"
        ]
        
        # Convert DataFrame to JSON serializable format
        try:
            import pandas as pd
            if isinstance(predictions, pd.DataFrame):
                # Convert DataFrame to list of dictionaries
                predictions_data = predictions.to_dict(orient='records')
            else:
                predictions_data = predictions
        except Exception as e:
            logger.error(f"Error converting predictions: {e}")
            predictions_data = str(predictions)  # Fallback to string
        
        # Create payload with just the predictions
        payload = {
            "admet": predictions_data
        }
        
        print(f"📤 SENDING to API:")
        print(f"   Simulation ID: {simulation_id}")
        print(f"   Payload size: {len(str(payload))} characters")
        print(f"   Predictions type: {type(predictions_data)}")
        if isinstance(predictions_data, list) and len(predictions_data) > 0:
            print(f"   First prediction sample: {predictions_data[0] if predictions_data else 'None'}")
        print(f"   Full payload preview: {str(payload)[:500]}{'...' if len(str(payload)) > 500 else ''}")
        
        success = False
        for url in urls_to_try:
            try:
                print(f"🔗 Trying URL: {url}")
                response = requests.put(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30
                )
                
                print(f"📨 API Response: {response.status_code}")
                if response.text:
                    print(f"📨 Response body: {response.text[:200]}{'...' if len(response.text) > 200 else ''}")
                
                if response.status_code == 200:
                    print(f"✅ Predictions sent for simulation {simulation_id}")
                    success = True
                    break
                else:
                    print(f"❌ API error {response.status_code} for simulation {simulation_id} at {url}")
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error for {url}: {e}")
                logger.error(f"Request failed for {url}: {e}")
                continue
            except Exception as e:
                print(f"❌ Unexpected error for {url}: {e}")
                logger.error(f"Unexpected error for {url}: {e}")
                continue
        
        if not success:
            print(f"❌ Failed to send predictions for simulation {simulation_id} to any endpoint")

    def predict_and_output(self, smiles_list):
        """
        Run model predictions and print ONLY the resulting data structure.
        """
        if not smiles_list:
            return

        # You can change this if you want JSON output always
        predictions = self.model.predict(smiles_list)

        # If the returned object is a DataFrame, convert to JSON for clean stdout
        try:
            import pandas as pd
            if isinstance(predictions, pd.DataFrame):
                # orient=records gives a list of dicts
                print(predictions.to_json(orient="records"))
            else:
                # Fallback: print raw
                print(predictions)
        except Exception:
            print(predictions)

    def callback(self, ch, method, properties, body):
        try:
            payload = body.decode('utf-8')
            print(f"📥 RECEIVED from AMQP: {payload}")
            
            simulation_id, smiles_list = self.parse_message_payload(payload)

            if not simulation_id or not smiles_list:
                # Invalid payload; ack to avoid poison loop
                logger.error("Invalid message format or missing data")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            print(f"🧪 Processing {len(smiles_list)} compounds for simulation {simulation_id}")
            print(f"🧬 SMILES to process: {smiles_list}")
            
            # Run ADMET predictions - get raw output
            predictions = self.model.predict(smiles_list)
            
            # Send raw predictions to API
            self.send_predictions_to_api(simulation_id, predictions)

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            # Log to stderr but do NOT print extra stuff to stdout
            logger.error(f"Prediction error: {e}")
            print(f"❌ Error processing message: {e}")
            # Decide: requeue or discard. Here we discard to avoid poison loop.
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.callback)
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
        except Exception:
            pass

def main():
    # Load from environment or fall back to literals
    AMQP_URL = os.getenv(
        "AMQP_URL",
        "amqps://zpjhgklu:18b44amdwCpc8ijdXIYgv-Lra0yfLCvx@dog.lmq.cloudamqp.com/zpjhgklu"
    )
    QUEUE_NAME = os.getenv("QUEUE_NAME", "test_queue")

    receiver = AMQPAdmetReceiver(AMQP_URL, QUEUE_NAME)
    receiver.connect()
    receiver.start()

if __name__ == "__main__":
    main()