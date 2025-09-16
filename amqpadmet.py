import os
import sys
import ssl
import json
import logging
from urllib.parse import urlparse

import pika

# QUIET LOGGING (only errors)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Import ADMET model (suppress possible model init chatter if needed)
from admet_ai import ADMETModel

class AMQPAdmetReceiver:
    def __init__(self, amqp_url: str, queue_name: str, batch_mode: bool = True):
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self.batch_mode = batch_mode
        self.model = None

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
            smiles_list = self.parse_smiles_payload(payload)

            if not smiles_list:
                # Nothing valid; ack to avoid poison loop
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            self.predict_and_output(smiles_list)

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            # Log to stderr but do NOT print extra stuff to stdout
            logger.error(f"Prediction error: {e}")
            # Decide: requeue or discard. Here we requeue once; you could add retry logic.
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