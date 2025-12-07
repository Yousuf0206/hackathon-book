# Chapter 2: Voice-to-Action Pipeline

## Learning Objectives
By the end of this chapter, you will be able to:
- Understand the components of voice-to-action systems for robotics
- Implement speech recognition and natural language processing pipelines
- Design intent parsing and entity extraction for robotic commands
- Create robust voice command validation and error handling
- Integrate voice processing with robotic action execution
- Evaluate and improve voice-to-action system performance

## 2.1 Introduction to Voice-to-Action Systems

Voice-to-action systems bridge the gap between human natural language and robotic action execution. These systems enable intuitive human-robot interaction by allowing users to issue commands in spoken natural language, which are then processed and translated into executable robotic actions.

### Key Components of Voice-to-Action Systems

A typical voice-to-action pipeline consists of several interconnected components:

1. **Audio Capture**: Recording and preprocessing audio input
2. **Speech Recognition**: Converting speech to text
3. **Natural Language Understanding (NLU)**: Interpreting the meaning of text
4. **Intent Classification**: Determining the user's goal
5. **Entity Extraction**: Identifying relevant objects, locations, and parameters
6. **Action Mapping**: Translating intents to robotic actions
7. **Validation and Safety Checks**: Ensuring safe and appropriate action execution
8. **Feedback Generation**: Providing confirmation or error messages

### Architecture Overview

```
[Microphone] → [Audio Processing] → [Speech Recognition] → [NLU] → [Intent Classification]
                                                                  ↓
[Speaker] ← [Audio Output] ← [Text-to-Speech] ← [Response Generation] ← [Action Mapping]
```

## 2.2 Audio Processing and Capture

### 2.2.1 Audio Input Pipeline

The audio processing pipeline begins with capturing and conditioning the audio signal:

```python
import pyaudio
import numpy as np
import webrtcvad
from scipy import signal
import threading
import queue

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, vad_aggressiveness=2):
        """
        Initialize audio processor with voice activity detection

        Args:
            sample_rate: Audio sampling rate (Hz)
            chunk_size: Size of audio chunks to process
            vad_agressiveness: VAD aggressiveness level (0-3)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # Audio stream configuration
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.frame_duration = 30  # ms
        self.frame_size = int(sample_rate * self.frame_duration / 1000)

        # Audio processing parameters
        self.energy_threshold = 1000  # Adjust based on environment
        self.silence_threshold = 500
        self.min_speech_frames = 10  # Minimum frames to consider speech

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_listening = False

    def start_audio_capture(self):
        """Start audio capture in a separate thread"""
        self.is_listening = True
        self.audio_thread = threading.Thread(target=self._capture_audio)
        self.audio_thread.start()

    def stop_audio_capture(self):
        """Stop audio capture"""
        self.is_listening = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()

    def _capture_audio(self):
        """Internal method to capture audio continuously"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )

        while self.is_listening:
            try:
                # Read audio frame
                frame_data = stream.read(self.frame_size, exception_on_overflow=False)

                # Convert to numpy array
                audio_frame = np.frombuffer(frame_data, dtype=np.int16)

                # Apply voice activity detection
                if self._is_voice_active(audio_frame):
                    # Add to processing queue
                    self.audio_queue.put(audio_frame)

            except Exception as e:
                print(f"Audio capture error: {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _is_voice_active(self, audio_frame):
        """Detect if voice is active in the audio frame"""
        # Calculate energy
        energy = np.sum(np.abs(audio_frame.astype(float))**2) / len(audio_frame)

        # Apply VAD
        try:
            vad_result = self.vad.is_speech(
                audio_frame.tobytes(),
                sample_rate=self.sample_rate
            )
        except:
            vad_result = False

        # Combine energy and VAD results
        return vad_result and energy > self.energy_threshold

    def get_audio_chunk(self, timeout=None):
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
```

### 2.2.2 Audio Preprocessing

Preprocessing improves speech recognition accuracy:

```python
from scipy import signal
import librosa

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def preprocess_audio(self, audio_data):
        """
        Apply preprocessing to audio data

        Args:
            audio_data: Raw audio samples

        Returns:
            Processed audio data ready for speech recognition
        """
        # Normalize audio
        normalized_audio = self._normalize(audio_data)

        # Apply noise reduction
        denoised_audio = self._reduce_noise(normalized_audio)

        # Apply bandpass filtering for speech frequencies
        filtered_audio = self._bandpass_filter(denoised_audio)

        # Apply pre-emphasis filter
        emphasized_audio = self._pre_emphasis(filtered_audio)

        return emphasized_audio

    def _normalize(self, audio_data):
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data.astype(float) / max_val
        return audio_data.astype(float)

    def _reduce_noise(self, audio_data, noise_duration=0.1):
        """Simple noise reduction by subtracting estimated noise floor"""
        noise_samples = int(noise_duration * self.sample_rate)
        if len(audio_data) > noise_samples:
            noise_floor = np.mean(np.abs(audio_data[:noise_samples]))
            return np.clip(audio_data - noise_floor, -1.0, 1.0)
        return audio_data

    def _bandpass_filter(self, audio_data, low_freq=300, high_freq=3400):
        """Apply bandpass filter to isolate speech frequencies"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        return filtered

    def _pre_emphasis(self, audio_data, alpha=0.97):
        """Apply pre-emphasis filter to boost high frequencies"""
        emphasized = np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
        return emphasized
```

## 2.3 Speech Recognition Systems

### 2.3.1 Integration with Speech Recognition APIs

Modern voice-to-action systems often leverage pre-trained speech recognition models:

```python
import speech_recognition as sr
import asyncio
from typing import Optional

class SpeechRecognizer:
    def __init__(self, provider="google", language="en-US"):
        """
        Initialize speech recognizer with specified provider

        Args:
            provider: Speech recognition provider ('google', 'wit', 'azure', etc.)
            language: Language code for recognition
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language
        self.provider = provider

        # Configure recognizer
        self.recognizer.energy_threshold = 3000  # Adjust sensitivity
        self.recognizer.dynamic_energy_threshold = True

        # Initialize provider-specific configurations
        self._setup_provider()

    def _setup_provider(self):
        """Configure provider-specific settings"""
        if self.provider == "google":
            # Google Cloud Speech-to-Text configuration
            self.api_key = self._load_api_key("google_speech")
        elif self.provider == "wit":
            # Wit.ai configuration
            self.api_key = self._load_api_key("wit_ai")
        elif self.provider == "local":
            # Local speech recognition (PocketSphinx)
            self.recognizer.pause_threshold = 0.8

    def _load_api_key(self, service_name):
        """Load API key from secure storage"""
        # Implementation would load from environment variables or secure storage
        import os
        return os.getenv(f"{service_name.upper()}_API_KEY")

    def recognize_speech(self, audio_data, timeout=5.0):
        """
        Recognize speech from audio data

        Args:
            audio_data: Audio data in the format expected by the recognizer
            timeout: Timeout for recognition attempt

        Returns:
            Recognized text or None if unsuccessful
        """
        try:
            # Create AudioData object
            audio = sr.AudioData(audio_data.tobytes(),
                               self.recognizer.energy_threshold,
                               2)  # Assuming 16-bit audio

            # Perform recognition based on provider
            if self.provider == "google":
                text = self.recognizer.recognize_google(audio, language=self.language)
            elif self.provider == "wit":
                text = self.recognizer.recognize_wit(audio, key=self.api_key)
            elif self.provider == "local":
                text = self.recognizer.recognize_sphinx(audio)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            return text.strip()

        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
```

### 2.3.2 Custom Speech Recognition Models

For specialized applications, custom models may be preferred:

```python
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class CustomSpeechRecognizer:
    def __init__(self, model_path="facebook/wav2vec2-base-960h"):
        """
        Initialize custom speech recognition model

        Args:
            model_path: Path to pretrained model or Hugging Face identifier
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def recognize_audio(self, audio_tensor, sampling_rate=16000):
        """
        Recognize speech using custom model

        Args:
            audio_tensor: Audio tensor (mono, 16kHz)
            sampling_rate: Audio sampling rate

        Returns:
            Recognized text
        """
        # Resample if necessary
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_tensor = resampler(audio_tensor)

        # Process audio
        inputs = self.processor(
            audio_tensor.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.lower()
```

## 2.4 Natural Language Understanding (NLU)

### 2.4.1 Intent Classification

Intent classification determines the user's goal from the recognized text:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

class IntentClassifier:
    def __init__(self, model_path="microsoft/DialoGPT-medium"):
        """
        Initialize intent classifier

        Args:
            model_path: Path to pretrained model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self._get_robot_intents())
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define robot-specific intents
        self.intents = self._get_robot_intents()

    def _get_robot_intents(self):
        """Define robot-specific intents"""
        return {
            "navigation_move_to_location": {
                "keywords": ["go to", "move to", "navigate to", "walk to", "drive to"],
                "slots": ["location"]
            },
            "manipulation_pick_object": {
                "keywords": ["pick up", "grab", "take", "lift", "get"],
                "slots": ["object", "location"]
            },
            "manipulation_place_object": {
                "keywords": ["place", "put", "set down", "release", "drop"],
                "slots": ["object", "location"]
            },
            "navigation_follow_person": {
                "keywords": ["follow", "track", "escort", "accompany"],
                "slots": ["person"]
            },
            "information_query": {
                "keywords": ["what", "where", "when", "who", "how", "tell me"],
                "slots": ["topic"]
            },
            "stop_action": {
                "keywords": ["stop", "halt", "pause", "cancel"],
                "slots": []
            }
        }

    def classify_intent(self, text):
        """
        Classify the intent of the given text

        Args:
            text: Input text to classify

        Returns:
            Dictionary with intent and confidence score
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities).item()

        intent_name = list(self.intents.keys())[predicted_class.item()]

        return {
            "intent": intent_name,
            "confidence": confidence,
            "slots": self._extract_slots(text, intent_name)
        }

    def _extract_slots(self, text, intent_name):
        """Extract named entities from text based on intent"""
        slots = {}
        intent_info = self.intents.get(intent_name, {})

        for slot_type in intent_info.get("slots", []):
            slot_value = self._extract_slot_value(text, slot_type)
            if slot_value:
                slots[slot_type] = slot_value

        return slots

    def _extract_slot_value(self, text, slot_type):
        """Extract specific slot value from text"""
        # Simple rule-based extraction - in practice, use NER models
        if slot_type == "location":
            # Look for location indicators
            location_indicators = ["to ", "at ", "in ", "on ", "near ", "by "]
            for indicator in location_indicators:
                if indicator in text.lower():
                    # Extract everything after the indicator
                    parts = text.lower().split(indicator)
                    if len(parts) > 1:
                        return parts[1].strip()
        elif slot_type == "object":
            # Look for object indicators
            object_indicators = ["the ", "a ", "an "]
            # Simple extraction based on keywords around object
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ["up", "down", "it", "that", "this"]:
                    if i + 1 < len(words):
                        return words[i + 1]
        elif slot_type == "person":
            # Look for person names or pronouns
            person_keywords = ["me", "you", "him", "her", "them", "john", "mary", "tom"]
            for keyword in person_keywords:
                if keyword in text.lower():
                    return keyword

        return None
```

### 2.4.2 Entity Extraction and Named Entity Recognition

Advanced entity extraction for robotic commands:

```python
import spacy
from typing import List, Dict, Tuple

class EntityExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize entity extractor with spaCy model

        Args:
            model_name: Name of spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install with: python -m spacy download {model_name}")
            raise

        # Define robot-specific entity patterns
        self.robot_entities = {
            "LOCATION": ["kitchen", "bedroom", "living room", "office", "garage", "bathroom",
                        "dining room", "hallway", "garden", "door", "window", "table", "chair"],
            "OBJECT": ["cup", "plate", "bottle", "box", "book", "phone", "keys", "wallet",
                      "ball", "toy", "food", "water", "medicine", "remote", "lamp"],
            "PERSON": ["mom", "dad", "brother", "sister", "son", "daughter", "grandma", "grandpa",
                      "teacher", "doctor", "friend", "neighbor"],
            "ACTION": ["pick", "place", "move", "grasp", "carry", "open", "close", "turn",
                      "press", "push", "pull", "lift", "lower", "raise"]
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using both spaCy and custom patterns

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping entity types to lists of extracted entities
        """
        doc = self.nlp(text)
        entities = {}

        # Extract standard NER entities
        standard_entities = {}
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "TIME", "ORDINAL", "CARDINAL"]:
                if ent.label_ not in standard_entities:
                    standard_entities[ent.label_] = []
                standard_entities[ent.label_].append(ent.text)

        # Extract robot-specific entities
        robot_entities = self._extract_robot_specific_entities(text)

        # Combine results
        entities.update(standard_entities)
        entities.update(robot_entities)

        return entities

    def _extract_robot_specific_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract robot-specific entities using custom patterns"""
        entities = {}

        text_lower = text.lower()

        for entity_type, entity_list in self.robot_entities.items():
            found_entities = []
            for entity in entity_list:
                if entity in text_lower:
                    found_entities.append(entity)
            if found_entities:
                entities[entity_type] = found_entities

        return entities

    def resolve_entity_references(self, text: str, context_entities: Dict) -> Dict:
        """
        Resolve pronouns and references in text using context

        Args:
            text: Input text with potential references
            context_entities: Previously identified entities

        Returns:
            Dictionary with resolved entities
        """
        doc = self.nlp(text)
        resolved_entities = {}

        for token in doc:
            if token.pos_ == "PRON":  # Pronoun
                # Resolve pronoun based on context
                resolved_entity = self._resolve_pronoun(token.text, context_entities)
                if resolved_entity:
                    resolved_entities[token.text] = resolved_entity
            elif token.text.lower() in ["it", "that", "this"]:
                # Resolve demonstrative pronouns
                resolved_entity = self._resolve_demonstrative(token.text, context_entities)
                if resolved_entity:
                    resolved_entities[token.text] = resolved_entity

        return resolved_entities

    def _resolve_pronoun(self, pronoun: str, context_entities: Dict) -> str:
        """Resolve a pronoun to its referent"""
        pronoun_lower = pronoun.lower()
        if pronoun_lower in ["it", "that", "this"]:
            # Look for the most recently mentioned object
            if "OBJECT" in context_entities and context_entities["OBJECT"]:
                return context_entities["OBJECT"][-1]
        elif pronoun_lower in ["him", "her", "them"]:
            # Look for the most recently mentioned person
            if "PERSON" in context_entities and context_entities["PERSON"]:
                return context_entities["PERSON"][-1]

        return None

    def _resolve_demonstrative(self, demonstrative: str, context_entities: Dict) -> str:
        """Resolve demonstrative pronouns"""
        if demonstrative.lower() in ["it", "that", "this"]:
            # Prefer objects, then locations
            if "OBJECT" in context_entities and context_entities["OBJECT"]:
                return context_entities["OBJECT"][-1]
            elif "LOCATION" in context_entities and context_entities["LOCATION"]:
                return context_entities["LOCATION"][-1]

        return None
```

## 2.5 Action Mapping and Execution

### 2.5.1 Intent-to-Action Translation

Mapping recognized intents to robotic actions:

```python
import rospy
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String
from typing import Dict, Any, Optional

class ActionMapper:
    def __init__(self, robot_interface):
        """
        Initialize action mapper for robot interface

        Args:
            robot_interface: Interface to the target robot
        """
        self.robot_interface = robot_interface
        self.action_templates = self._define_action_templates()

    def _define_action_templates(self):
        """Define action templates for different intents"""
        return {
            "navigation_move_to_location": self._handle_navigation,
            "manipulation_pick_object": self._handle_manipulation_pick,
            "manipulation_place_object": self._handle_manipulation_place,
            "navigation_follow_person": self._handle_follow_person,
            "information_query": self._handle_information_query,
            "stop_action": self._handle_stop_action
        }

    def map_and_execute(self, intent_result: Dict[str, Any]) -> bool:
        """
        Map intent to action and execute

        Args:
            intent_result: Result from intent classification including slots

        Returns:
            True if action was successfully mapped and initiated
        """
        intent_name = intent_result["intent"]
        slots = intent_result.get("slots", {})
        confidence = intent_result.get("confidence", 0.0)

        # Validate confidence threshold
        if confidence < 0.7:  # Threshold for reliable execution
            print(f"Intent confidence {confidence:.2f} below threshold, not executing")
            return False

        # Check if intent handler exists
        if intent_name not in self.action_templates:
            print(f"No handler for intent: {intent_name}")
            return False

        # Execute action
        try:
            success = self.action_templates[intent_name](slots)
            return success
        except Exception as e:
            print(f"Error executing action for intent {intent_name}: {e}")
            return False

    def _handle_navigation(self, slots: Dict[str, Any]) -> bool:
        """Handle navigation intent"""
        if "location" not in slots:
            print("No location specified for navigation")
            return False

        location = slots["location"]

        # Convert location to coordinates (this would use a map)
        target_pose = self._location_to_pose(location)
        if target_pose is None:
            print(f"Unknown location: {location}")
            return False

        # Execute navigation
        return self.robot_interface.navigate_to_pose(target_pose)

    def _handle_manipulation_pick(self, slots: Dict[str, Any]) -> bool:
        """Handle object pick intent"""
        if "object" not in slots:
            print("No object specified for picking")
            return False

        obj = slots["object"]

        # Find object in environment
        object_pose = self.robot_interface.find_object(obj)
        if object_pose is None:
            print(f"Object '{obj}' not found")
            return False

        # Execute pick action
        return self.robot_interface.pick_object(object_pose)

    def _handle_manipulation_place(self, slots: Dict[str, Any]) -> bool:
        """Handle object placement intent"""
        if "object" not in slots or "location" not in slots:
            print("Need both object and location for placement")
            return False

        obj = slots["object"]
        location = slots["location"]

        # Find placement location
        target_pose = self._location_to_pose(location)
        if target_pose is None:
            print(f"Unknown location: {location}")
            return False

        # Execute place action
        return self.robot_interface.place_object(target_pose)

    def _handle_follow_person(self, slots: Dict[str, Any]) -> bool:
        """Handle person following intent"""
        person = slots.get("person", "unknown")

        # Start following person
        return self.robot_interface.follow_person(person)

    def _handle_information_query(self, slots: Dict[str, Any]) -> bool:
        """Handle information query intent"""
        topic = slots.get("topic", "unknown")

        # Query robot knowledge base
        response = self.robot_interface.query_knowledge(topic)

        # Speak response
        self.robot_interface.speak(response)
        return True

    def _handle_stop_action(self, slots: Dict[str, Any]) -> bool:
        """Handle stop action intent"""
        return self.robot_interface.stop_current_action()

    def _location_to_pose(self, location: str) -> Optional[Pose]:
        """Convert location name to pose coordinates"""
        # This would typically use a map or predefined locations
        location_map = {
            "kitchen": Pose(position=Point(x=2.0, y=1.0, z=0.0)),
            "bedroom": Pose(position=Point(x=-1.0, y=3.0, z=0.0)),
            "living room": Pose(position=Point(x=0.0, y=0.0, z=0.0)),
            "office": Pose(position=Point(x=3.0, y=-2.0, z=0.0)),
        }

        return location_map.get(location.lower())
```

### 2.5.2 Safety Validation and Error Handling

Critical safety checks for voice commands:

```python
class SafetyValidator:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.forbidden_commands = [
            "hurt", "damage", "break", "destroy", "attack", "hit",
            "injure", "kill", "dangerous", "unsafe", "harm"
        ]

    def validate_action(self, intent_result: Dict[str, Any], original_text: str) -> bool:
        """
        Validate action for safety before execution

        Args:
            intent_result: Parsed intent and slots
            original_text: Original command text

        Returns:
            True if action is safe to execute
        """
        # Check for forbidden words
        if self._contains_forbidden_words(original_text):
            print("Command contains forbidden words, rejecting")
            return False

        # Validate intent-specific safety checks
        intent = intent_result["intent"]
        slots = intent_result.get("slots", {})

        if intent == "navigation_move_to_location":
            return self._validate_navigation(slots)
        elif intent.startswith("manipulation"):
            return self._validate_manipulation(intent, slots)
        elif intent == "navigation_follow_person":
            return self._validate_follow(slots)

        return True

    def _contains_forbidden_words(self, text: str) -> bool:
        """Check if text contains forbidden words"""
        text_lower = text.lower()
        for forbidden_word in self.forbidden_commands:
            if forbidden_word in text_lower:
                return True
        return False

    def _validate_navigation(self, slots: Dict[str, Any]) -> bool:
        """Validate navigation command safety"""
        if "location" not in slots:
            return False

        location = slots["location"]

        # Check if location is safe
        if location.lower() in ["stairs", "roof", "attic", "basement"]:
            # Additional safety checks for potentially dangerous locations
            if not self._is_safe_to_navigate(location):
                return False

        return True

    def _validate_manipulation(self, intent: str, slots: Dict[str, Any]) -> bool:
        """Validate manipulation command safety"""
        if "object" not in slots:
            return False

        obj = slots["object"]

        # Check if object is safe to manipulate
        if obj.lower() in ["knife", "blade", "weapon", "fire", "hot", "sharp"]:
            return False

        return True

    def _validate_follow(self, slots: Dict[str, Any]) -> bool:
        """Validate follow command safety"""
        person = slots.get("person", "")

        # Check if following is safe
        if person.lower() in ["unknown", "stranger", "nobody"]:
            return False

        return True

    def _is_safe_to_navigate(self, location: str) -> bool:
        """Check if navigation to location is safe"""
        # This would check robot's environment awareness
        # For now, assume all locations are safe if not explicitly dangerous
        return True
```

## 2.6 Voice Command Feedback System

### 2.6.1 Text-to-Speech Response Generation

Generating appropriate feedback for voice commands:

```python
import pyttsx3
import threading
from typing import Dict, Any

class VoiceFeedbackSystem:
    def __init__(self, rate=150, volume=0.9):
        """
        Initialize voice feedback system

        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)  # Use first available voice

        self.feedback_lock = threading.Lock()
        self.pending_speech = []

    def generate_response(self, intent_result: Dict[str, Any], success: bool) -> str:
        """
        Generate appropriate response based on intent and execution result

        Args:
            intent_result: Parsed intent result
            success: Whether action was successful

        Returns:
            Response text to speak
        """
        intent = intent_result["intent"]
        confidence = intent_result.get("confidence", 0.0)

        if confidence < 0.5:
            return "I heard something but I'm not sure what you meant. Could you repeat that?"
        elif not success:
            return self._generate_failure_response(intent_result)
        else:
            return self._generate_success_response(intent_result)

    def _generate_success_response(self, intent_result: Dict[str, Any]) -> str:
        """Generate response for successful action"""
        intent = intent_result["intent"]

        if intent == "navigation_move_to_location":
            location = intent_result.get("slots", {}).get("location", "target location")
            return f"Okay, I'm navigating to the {location}."
        elif intent.startswith("manipulation_pick"):
            obj = intent_result.get("slots", {}).get("object", "object")
            return f"Okay, I'm picking up the {obj}."
        elif intent.startswith("manipulation_place"):
            obj = intent_result.get("slots", {}).get("object", "object")
            location = intent_result.get("slots", {}).get("location", "location")
            return f"Okay, I'm placing the {obj} at the {location}."
        elif intent == "navigation_follow_person":
            person = intent_result.get("slots", {}).get("person", "person")
            return f"Okay, I'm following {person}."
        elif intent == "stop_action":
            return "Okay, stopping current action."
        else:
            return "Okay, I've completed the requested action."

    def _generate_failure_response(self, intent_result: Dict[str, Any]) -> str:
        """Generate response for failed action"""
        intent = intent_result["intent"]

        if intent == "navigation_move_to_location":
            location = intent_result.get("slots", {}).get("location", "location")
            return f"I couldn't navigate to the {location}. Is it accessible?"
        elif intent.startswith("manipulation_pick"):
            obj = intent_result.get("slots", {}).get("object", "object")
            return f"I couldn't find or pick up the {obj}."
        elif intent.startswith("manipulation_place"):
            obj = intent_result.get("slots", {}).get("object", "object")
            location = intent_result.get("slots", {}).get("location", "location")
            return f"I couldn't place the {obj} at the {location}."
        else:
            return "Sorry, I couldn't complete that action. Could you try again?"

    def speak(self, text: str, blocking: bool = False):
        """
        Speak the given text

        Args:
            text: Text to speak
            blocking: Whether to wait for speech to complete
        """
        with self.feedback_lock:
            self.engine.say(text)
            if blocking:
                self.engine.runAndWait()
            else:
                # Run in background
                threading.Thread(target=self._speak_non_blocking, args=(text,)).start()

    def _speak_non_blocking(self, text: str):
        """Speak text in a non-blocking manner"""
        self.engine.say(text)
        self.engine.runAndWait()

    def stop_speaking(self):
        """Stop current speech"""
        self.engine.stop()
```

## 2.7 Complete Voice-to-Action System Integration

### 2.7.1 Main VLA Voice Processing Pipeline

Bringing all components together:

```python
class VLAVoiceSystem:
    def __init__(self, robot_interface, provider="google"):
        """
        Initialize complete VLA voice-to-action system

        Args:
            robot_interface: Interface to the target robot
            provider: Speech recognition provider
        """
        self.audio_processor = AudioProcessor()
        self.speech_recognizer = SpeechRecognizer(provider=provider)
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.action_mapper = ActionMapper(robot_interface)
        self.safety_validator = SafetyValidator(robot_interface)
        self.voice_feedback = VoiceFeedbackSystem()

        self.is_active = False

    def start_system(self):
        """Start the voice processing system"""
        self.is_active = True
        self.audio_processor.start_audio_capture()
        print("VLA Voice System started. Listening for commands...")

    def stop_system(self):
        """Stop the voice processing system"""
        self.is_active = False
        self.audio_processor.stop_audio_capture()
        print("VLA Voice System stopped.")

    def process_voice_command(self, timeout=10.0):
        """
        Process a single voice command from capture to execution

        Args:
            timeout: Maximum time to wait for command

        Returns:
            Tuple of (success, response_text)
        """
        if not self.is_active:
            return False, "System is not active"

        # Capture audio
        print("Listening...")
        audio_chunks = []
        silence_count = 0
        max_silence = 5  # Number of silent chunks to stop

        start_time = time.time()
        while self.is_active and time.time() - start_time < timeout:
            chunk = self.audio_processor.get_audio_chunk(timeout=0.1)
            if chunk is not None:
                audio_chunks.append(chunk)
                silence_count = 0  # Reset silence counter
            else:
                silence_count += 1
                if silence_count > max_silence and len(audio_chunks) > 0:
                    break  # Stop after period of silence

        if not audio_chunks:
            return False, "No audio captured"

        # Combine audio chunks
        combined_audio = np.concatenate(audio_chunks)

        # Preprocess audio
        processed_audio = AudioPreprocessor().preprocess_audio(combined_audio)

        # Recognize speech
        recognized_text = self.speech_recognizer.recognize_speech(processed_audio)
        if not recognized_text:
            response = "I couldn't understand what you said. Could you repeat that?"
            self.voice_feedback.speak(response)
            return False, response

        print(f"Heard: {recognized_text}")

        # Classify intent
        intent_result = self.intent_classifier.classify_intent(recognized_text)
        print(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")

        # Extract entities
        entities = self.entity_extractor.extract_entities(recognized_text)
        print(f"Entities: {entities}")

        # Add entities to intent result
        intent_result["entities"] = entities

        # Validate safety
        if not self.safety_validator.validate_action(intent_result, recognized_text):
            response = "That command seems unsafe. I can't execute it."
            self.voice_feedback.speak(response)
            return False, response

        # Map and execute action
        success = self.action_mapper.map_and_execute(intent_result)

        # Generate and speak response
        response = self.voice_feedback.generate_response(intent_result, success)
        self.voice_feedback.speak(response)

        return success, response

    def continuous_listening_loop(self):
        """Run continuous listening and command processing"""
        print("Starting continuous listening loop...")

        self.start_system()

        try:
            while self.is_active:
                success, response = self.process_voice_command(timeout=5.0)
                if not success:
                    print(f"Command failed: {response}")

        except KeyboardInterrupt:
            print("\nStopping voice system...")
        finally:
            self.stop_system()
```

## 2.8 Performance Optimization and Tuning

### 2.8.1 Configuration and Tuning Parameters

Key parameters for optimizing voice-to-action performance:

```python
class VoiceSystemConfig:
    """Configuration class for voice system parameters"""

    def __init__(self):
        # Audio processing parameters
        self.audio_sample_rate = 16000
        self.audio_chunk_size = 1024
        self.vad_aggressiveness = 2
        self.energy_threshold = 1000

        # Recognition parameters
        self.recognition_timeout = 5.0
        self.confidence_threshold = 0.7

        # NLU parameters
        self.intent_confidence_threshold = 0.6
        self.max_context_length = 10  # Number of previous interactions to remember

        # Safety parameters
        self.safety_check_enabled = True
        self.response_timeout = 30.0  # Maximum time for action completion

        # Performance parameters
        self.max_concurrent_requests = 3
        self.cache_size = 100  # Number of recent commands to cache
```

## 2.9 Testing and Validation

### 2.9.1 Unit Tests for Voice Components

Testing individual components of the voice-to-action pipeline:

```python
import unittest
from unittest.mock import Mock, patch

class TestVoiceToAction(unittest.TestCase):
    def setUp(self):
        self.robot_mock = Mock()
        self.voice_system = VLAVoiceSystem(self.robot_mock)

    def test_audio_processing(self):
        """Test audio processing pipeline"""
        processor = AudioProcessor()
        test_audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)

        processed = AudioPreprocessor().preprocess_audio(test_audio)

        # Check that output is normalized
        self.assertTrue(np.all(processed >= -1.0) and np.all(processed <= 1.0))

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_speech_recognition(self, mock_recognize):
        """Test speech recognition component"""
        mock_recognize.return_value = "move to kitchen"

        # This would test the speech recognition integration
        # Implementation depends on actual audio data format

    def test_intent_classification(self):
        """Test intent classification"""
        classifier = IntentClassifier()

        # Test a navigation command
        result = classifier.classify_intent("go to the kitchen")

        self.assertIn(result["intent"], classifier.intents.keys())
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_safety_validation(self):
        """Test safety validation"""
        validator = SafetyValidator(self.robot_mock)

        # Test safe command
        safe_intent = {"intent": "navigation_move_to_location", "slots": {"location": "kitchen"}}
        safe_text = "go to the kitchen"
        self.assertTrue(validator.validate_action(safe_intent, safe_text))

        # Test unsafe command
        unsafe_intent = {"intent": "navigation_move_to_location", "slots": {"location": "kitchen"}}
        unsafe_text = "go hurt yourself in the kitchen"
        self.assertFalse(validator.validate_action(unsafe_intent, unsafe_text))

if __name__ == '__main__':
    unittest.main()
```

## Summary

The voice-to-action pipeline is a critical component of VLA robotics systems, enabling natural and intuitive human-robot interaction. This chapter covered the complete pipeline from audio capture and speech recognition to intent classification, action mapping, and safety validation. Key components include robust audio processing, accurate speech recognition, sophisticated natural language understanding, and comprehensive safety mechanisms.

The implementation provided demonstrates a production-ready voice-to-action system with proper error handling, safety validation, and feedback mechanisms. Future chapters will build upon this foundation to create more sophisticated cognitive planning and multi-modal perception systems.

## Exercises

1. Implement a custom keyword spotting system that triggers the voice-to-action pipeline when specific wake words are detected.

2. Extend the entity extraction system to handle spatial relationships (e.g., "the cup on the table").

3. Design a confidence-based fallback mechanism that asks for clarification when speech recognition confidence is low.

4. Create a voice command logging system that tracks command success rates and identifies common failure patterns.

5. Implement speaker identification to personalize responses based on who is speaking to the robot.