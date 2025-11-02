import numpy as np
from typing import Dict, List, Any
from copy import deepcopy


class SensorModel:
    """
    Basis Sensor Model - Fügt Unsicherheiten zu Ground Truth hinzu
    
    Dies ist das KRITISCHE Modul für stochastisches Verhalten!
    """
    
    def __init__(self, config):  # SensorConfig type hint entfernt für jetzt
        self.config = config
        self.detection_error_rate = config.detection_error_rate
        self.false_positive_rate = config.false_positive_rate
        self.distance_noise_std = config.distance_noise_std
        self.velocity_noise_std = config.velocity_noise_std
        self.latency = config.latency
        
        # Latenz-Buffer für zeitverzögerte Messungen
        self.observation_buffer = []
        
    def perceive(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hauptfunktion: Ground Truth → Noisy Observation
        
        Args:
            ground_truth: Perfekte Sensordaten
            
        Returns:
            noisy_observation: Verrauschte Sensordaten
        """
        noisy_obs = {
            'ego': deepcopy(ground_truth['ego']),  # ← deepcopy statt .copy()
            'vehicles': [],
            'detection_info': {
                'missed_vehicles': [],
                'false_positives': [],
                'sensor_quality': 1.0 - self.detection_error_rate
            },
            'timestamp': ground_truth['timestamp']
        }
        
        # 1. Detektionsfehler und Rauschen für jedes Fahrzeug
        for vehicle in ground_truth['vehicles']:
            # STOCHASTISCH: Wird Fahrzeug erkannt?
            if np.random.rand() < self.detection_error_rate:
                # FALSCH-NEGATIV: Fahrzeug nicht erkannt!
                noisy_obs['detection_info']['missed_vehicles'].append(vehicle['id'])
                continue
            
            # Fahrzeug wurde erkannt - jetzt Rauschen hinzufügen
            noisy_vehicle = self._add_noise(vehicle)
            noisy_obs['vehicles'].append(noisy_vehicle)
        
        # 2. False Positives (Geisterfahrzeuge)
        if np.random.rand() < self.false_positive_rate:
            ghost = self._generate_ghost_vehicle(ground_truth)
            noisy_obs['vehicles'].append(ghost)
            noisy_obs['detection_info']['false_positives'].append(ghost['id'])
        
        # 3. Latenz simulieren (optional)
        if self.latency > 0:
            noisy_obs = self._apply_latency(noisy_obs)
        
        return noisy_obs
    
    def _add_noise(self, vehicle: Dict) -> Dict:
        """Fügt Gauß-Rauschen zu Messungen hinzu"""
        noisy_vehicle = deepcopy(vehicle)  # ← deepcopy statt .copy()
        
        # Distanz-Rauschen
        true_distance = np.sqrt(
            vehicle['relative_longitudinal']**2 + 
            vehicle['relative_lateral']**2
        )
        noise_distance = np.random.normal(0, self.distance_noise_std)
        noisy_distance = max(0, true_distance + noise_distance)
        
        # Skaliere relative Koordinaten proportional
        if true_distance > 0:
            scale = noisy_distance / true_distance
            noisy_vehicle['relative_longitudinal'] *= scale
            noisy_vehicle['relative_lateral'] *= scale
        
        # Geschwindigkeits-Rauschen
        noise_velocity = np.random.normal(0, self.velocity_noise_std)
        noisy_vehicle['velocity'] = max(0, vehicle['velocity'] + noise_velocity)
        
        # Confidence Score (höher wenn Detektion klar)
        noisy_vehicle['confidence'] = 1.0 - self.detection_error_rate
        noisy_vehicle['sensor_source'] = 'radar'  # Kann erweitert werden
        
        return noisy_vehicle
    
    def _generate_ghost_vehicle(self, ground_truth: Dict) -> Dict:
        """Generiert False Positive (Geisterfahrzeug)"""
        # Zufällige Position in der Nähe
        ghost = {
            'id': f'ghost_{np.random.randint(1000, 9999)}',
            'position': (
                ground_truth['ego']['position'][0] + np.random.uniform(20, 80),
                ground_truth['ego']['position'][1] + np.random.uniform(-8, 8)
            ),
            'velocity': np.random.uniform(20, 35),
            'relative_longitudinal': np.random.uniform(20, 80),
            'relative_lateral': np.random.uniform(-8, 8),
            'confidence': 0.5,  # Niedrig weil False Positive
            'sensor_source': 'radar',
            'is_ghost': True  # Marker für Debugging
        }
        return ghost
    
    def _apply_latency(self, observation: Dict) -> Dict:
        """Simuliert Sensor-Latenz"""
        self.observation_buffer.append(deepcopy(observation))  # ← deepcopy hinzugefügt
        
        # Latenz in Zeitschritten (bei 10 Hz = 0.1s pro Schritt)
        latency_steps = int(self.latency / 0.1)
        
        if len(self.observation_buffer) > latency_steps:
            delayed_obs = self.observation_buffer.pop(0)
            return delayed_obs
        else:
            # Noch nicht genug Buffer - gebe aktuelles zurück
            return observation