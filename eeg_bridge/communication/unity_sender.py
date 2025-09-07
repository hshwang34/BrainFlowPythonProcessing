"""
Unity communication interface

This module handles UDP communication with Unity applications,
formatting EEG analysis results into JSON messages.
"""

import json
import logging
import socket
from typing import Dict, Any

from ..core.data_types import MentalState
from ..core.config import UDP_HOST, UDP_PORT


class UnitySender:
    """
    Send mental state updates to Unity via UDP JSON messages
    
    This class handles the communication with Unity, formatting the EEG
    analysis results into JSON and sending them over UDP.
    """
    
    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self._setup_socket()
    
    def _setup_socket(self):
        """Setup UDP socket for communication"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logging.info(f"UDP sender initialized: {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to setup UDP socket: {e}")
    
    def send_state(self, state: MentalState) -> bool:
        """
        Send mental state to Unity
        
        Args:
            state: Current mental state
            
        Returns:
            bool: True if sent successfully
        """
        if self.socket is None:
            return False
            
        try:
            # Create JSON message
            message = {
                "t": state.timestamp,
                "alpha": float(state.band_powers.alpha),
                "beta": float(state.band_powers.beta),
                "theta": float(state.band_powers.theta),
                "relax_index": float(state.relax_index),
                "focus_index": float(state.focus_index),
                "state": state.state,
                "mi_prob_left": float(state.mi_prob_left),
                "mi_prob_right": float(state.mi_prob_right),
                "artifact": state.artifact
            }
            
            # Send JSON over UDP
            json_str = json.dumps(message)
            self.socket.sendto(json_str.encode('utf-8'), (self.host, self.port))
            return True
            
        except Exception as e:
            logging.error(f"Failed to send UDP message: {e}")
            return False
    
    def close(self):
        """Close UDP socket"""
        if self.socket:
            self.socket.close()
