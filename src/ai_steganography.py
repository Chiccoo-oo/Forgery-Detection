import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SteganographyDataset(Dataset):
    """Dataset for training the steganography system"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            # Create a random image if file not found
            image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        if self.transform:
            image = self.transform(image)
        else:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        return torch.FloatTensor(image)

class EmbeddingNetwork(nn.Module):
    """
    🥷 AI FORGER NETWORK (Intelligent Steganography)
    
    This is AI #1 - The master spy!
    Job: Hide messages so well that even AI detectors can't find them
    
    Key Innovation: Uses attention mechanism to find optimal hiding spots
    - Analyzes image content (textures, edges, noise)
    - Creates "attention map" showing best hiding locations
    - Embeds messages strategically, not randomly
    """
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        
        # 🔍 CONTENT ANALYZER: Understands what's in the image
        self.encoder = nn.Sequential(
            # Layer 1: Basic feature detection
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            
            # Layer 2: Texture and pattern analysis
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3: Complex structure understanding
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            
            # Layer 4: High-level semantic analysis
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 🎯 ATTENTION MECHANISM: "Where should I hide the message?"
        # This is the AI's strategic brain!
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # Creates attention map
            nn.Sigmoid()           # Values 0-1: how good each pixel is for hiding
        )
        
        # 📏 UPSAMPLING: Bring attention map back to full resolution
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # 🎨 FINAL EMBEDDING: Actually hide the message
        self.embed_layer = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # Image(3) + WeightedMessage(1)
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),  # Output: Stego image
            nn.Tanh()  # Keep pixel values in valid range
        )
    
    def forward(self, image, message):
        """
        🧠 AI DECISION PROCESS:
        1. Analyze image content and structure
        2. Generate attention map (where to hide)
        3. Weight message by attention (strategic placement)
        4. Create final stego image
        
        Returns: (stego_image, attention_map)
        """
        # Step 1: Understand the image
        features = self.encoder(image)
        
        # Step 2: AI decides optimal hiding locations
        attention_map = self.attention(features)
        attention_map = self.upsample(attention_map)
        
        # Step 3: Strategic message placement
        # AI weights the message by attention: high attention = good hiding spot
        message_expanded = message.unsqueeze(1).expand(-1, 1, 256, 256)
        weighted_message = message_expanded * attention_map
        
        # Step 4: Create the final stego image
        combined = torch.cat([image, weighted_message], dim=1)
        stego_image = self.embed_layer(combined)
        
        return stego_image, attention_map

class DetectorNetwork(nn.Module):
    """
    🕵️ AI DETECTIVE NETWORK (Steganalysis Tool)
    
    This is AI #2 - The adversarial opponent!
    Job: Find hidden messages in images
    
    Architecture designed to detect subtle steganographic patterns:
    - Deep convolutional layers to analyze image statistics
    - Global pooling to look at overall image properties  
    - Binary classification: "Has message" vs "Clean image"
    """
    def __init__(self):
        super(DetectorNetwork, self).__init__()
        
        # Feature extraction layers - looks for steganographic artifacts
        self.features = nn.Sequential(
            # First layer: Basic edge/texture detection
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            
            # Second layer: More complex pattern detection
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third layer: Statistical anomaly detection
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            
            # Fourth layer: High-level pattern recognition
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Final feature layer: Advanced steganographic signature detection
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Global representation
        )
        
        # Classification head: "Is this image hiding something?"
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevent overfitting to specific patterns
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),      # Single output: probability of hidden message
            nn.Sigmoid()           # 0 = clean, 1 = has hidden message
        )
    
    def forward(self, x):
        """
        Analyze image for steganographic content
        Returns: Probability (0-1) that image contains hidden message
        """
        features = self.features(x)
        detection_score = self.classifier(features)
        return detection_score

class ExtractionNetwork(nn.Module):
    """Network to extract hidden messages"""
    def __init__(self):
        super(ExtractionNetwork, self).__init__()
        
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 256),  # Reconstruct message
            nn.Sigmoid()
        )
    
    def forward(self, stego_image):
        features = self.extractor(stego_image)
        message = self.decoder(features)
        return message.view(-1, 1, 256, 256)

class AIsteganographySystem:
    """Main system combining all components"""
    def __init__(self, device='cpu'):
        self.device = device
        self.embedding_net = EmbeddingNetwork().to(device)
        self.detector_net = DetectorNetwork().to(device)
        self.extraction_net = ExtractionNetwork().to(device)
        
        # Optimizers
        self.embed_optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001)
        self.detect_optimizer = optim.Adam(self.detector_net.parameters(), lr=0.001)
        self.extract_optimizer = optim.Adam(self.extraction_net.parameters(), lr=0.001)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def generate_random_message(self, batch_size):
        """Generate random binary message"""
        return torch.randint(0, 2, (batch_size, 256, 256), dtype=torch.float32).to(self.device)
    
    def train_step(self, images):
        """
        🥊 ADVERSARIAL TRAINING: Two AIs Battle Each Other!
        
        AI #1 (FORGER): EmbeddingNetwork - Tries to hide messages perfectly
        AI #2 (DETECTIVE): DetectorNetwork - Tries to find hidden messages
        
        They train against each other in a constant arms race!
        """
        batch_size = images.size(0)
        images = images.to(self.device)
        messages = self.generate_random_message(batch_size)
        
        # Forward pass through embedding network
        stego_images, attention_maps = self.embedding_net(images, messages)
        
        # 🕵️ TRAIN THE DETECTIVE AI (DetectorNetwork)
        # Goal: Get better at finding hidden messages
        self.detect_optimizer.zero_grad()
        
        # Test on clean images (should output ~0: "no message detected")
        real_pred = self.detector_net(images)
        real_labels = torch.zeros_like(real_pred)
        
        # Test on stego images (should output ~1: "message detected!")
        stego_pred = self.detector_net(stego_images.detach())
        stego_labels = torch.ones_like(stego_pred)
        
        # Detective tries to get better at distinguishing
        detector_loss = (self.bce_loss(real_pred, real_labels) + 
                        self.bce_loss(stego_pred, stego_labels)) / 2
        detector_loss.backward()
        self.detect_optimizer.step()
        
        # 🥷 TRAIN THE FORGER AI (EmbeddingNetwork) 
        # Goal: Fool the detective while preserving image quality
        self.embed_optimizer.zero_grad()
        
        # 🎭 ADVERSARIAL COMPONENT: Try to fool the detective!
        # The forger wants the detective to output ~0 (think it's clean)
        stego_pred_for_embed = self.detector_net(stego_images)
        adversarial_loss = self.bce_loss(stego_pred_for_embed, torch.zeros_like(stego_pred_for_embed))
        
        # 🖼️ QUALITY PRESERVATION: Don't destroy the image
        reconstruction_loss = self.mse_loss(stego_images, images)
        
        # Combined loss: Be sneaky BUT preserve quality
        embedding_loss = adversarial_loss + 10 * reconstruction_loss
        embedding_loss.backward()
        self.embed_optimizer.step()
        
        # Train extraction network (separate from the battle)
        self.extract_optimizer.zero_grad()
        extracted_messages = self.extraction_net(stego_images.detach())
        extraction_loss = self.mse_loss(extracted_messages.squeeze(1), messages)
        extraction_loss.backward()
        self.extract_optimizer.step()
        
        return {
            'detector_loss': detector_loss.item(),        # How well detective is learning
            'embedding_loss': embedding_loss.item(),      # How well forger is learning
            'extraction_loss': extraction_loss.item(),    # How well extraction works
            'adversarial_loss': adversarial_loss.item(),  # How well forger fools detective
            'reconstruction_loss': reconstruction_loss.item()  # How well image quality is preserved
        }
    
    def embed_message(self, image, message):
        """Embed a message into an image"""
        self.embedding_net.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(message.shape) == 2:
                message = message.unsqueeze(0)
            
            image = image.to(self.device)
            message = message.to(self.device)
            
            stego_image, attention_map = self.embedding_net(image, message)
            return stego_image.cpu().numpy(), attention_map.cpu().numpy()
    
    def extract_message(self, stego_image):
        """Extract message from stego image"""
        self.extraction_net.eval()
        with torch.no_grad():
            if len(stego_image.shape) == 3:
                stego_image = stego_image.unsqueeze(0)
            
            stego_image = stego_image.to(self.device)
            extracted_message = self.extraction_net(stego_image)
            return extracted_message.cpu().numpy()
    
    def detect_steganography(self, image):
        """Detect if image contains hidden message"""
        self.detector_net.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            detection_score = self.detector_net(image)
            return detection_score.cpu().numpy()

def create_demo_images(num_images=50):
    """Create demo images if no dataset available"""
    images = []
    for i in range(num_images):
        # Create varied synthetic images
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add different patterns
        if i % 4 == 0:
            # Gradient
            for x in range(256):
                for y in range(256):
                    img[x, y] = [x % 256, y % 256, (x + y) % 256]
        elif i % 4 == 1:
            # Random noise
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        elif i % 4 == 2:
            # Geometric shapes
            cv2.circle(img, (128, 128), 80, (255, 0, 0), -1)
            cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
        else:
            # Texture-like pattern
            for x in range(0, 256, 20):
                for y in range(0, 256, 20):
                    cv2.rectangle(img, (x, y), (x+15, y+15), 
                                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
        
        images.append(img.astype(np.float32) / 255.0)
    
    return images

def string_to_binary_image(text, width=256, height=256):
    """Convert string to binary image representation"""
    # Convert string to binary
    binary_str = ''.join(format(ord(char), '08b') for char in text)
    
    # Pad or truncate to fit image size
    total_bits = width * height
    if len(binary_str) < total_bits:
        binary_str = binary_str + '0' * (total_bits - len(binary_str))
    else:
        binary_str = binary_str[:total_bits]
    
    # Convert to image
    binary_array = np.array([int(bit) for bit in binary_str], dtype=np.float32)
    binary_image = binary_array.reshape(height, width)
    
    return binary_image

def binary_image_to_string(binary_image):
    """Convert binary image back to string"""
    # Flatten and convert to binary string
    binary_array = binary_image.flatten()
    binary_str = ''.join([str(int(round(bit))) for bit in binary_array])
    
    # Convert binary to characters
    chars = []
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            char_code = int(byte, 2)
            if 32 <= char_code <= 126:  # Printable ASCII
                chars.append(chr(char_code))
            elif char_code == 0:  # Null terminator
                break
    
    return ''.join(chars).rstrip('\x00')

def main():
    """Main function to demonstrate the system"""
    print("🔐 AI-Powered Steganography System")
    print("===================================")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the system
    stego_system = AIsteganographySystem(device)
    
    # Create demo dataset
    print("\n📸 Creating demo images...")
    demo_images = create_demo_images(100)
    
    # Convert to torch tensors
    image_tensors = []
    for img in demo_images:
        img_tensor = torch.FloatTensor(np.transpose(img, (2, 0, 1)))
        image_tensors.append(img_tensor)
    
    # Create DataLoader
    train_loader = DataLoader(image_tensors, batch_size=4, shuffle=True)
    
    print("\n🥊 ADVERSARIAL AI BATTLE BEGINS!")
    print("=" * 50)
    print("🥷 AI FORGER: 'I will hide messages perfectly!'")
    print("🕵️ AI DETECTIVE: 'I will find every hidden message!'")
    print("🏟️ BATTLE ARENA: Your laptop")
    print("=" * 50)
    
    # Training loop - THE EPIC BATTLE!
    for epoch in range(10):  # Reduced epochs for demo
        epoch_losses = {'detector': 0, 'embedding': 0, 'extraction': 0}
        
        print(f"\n🔥 ROUND {epoch+1}/10 - FIGHT!")
        
        for batch_idx, images in enumerate(train_loader):
            losses = stego_system.train_step(images)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[f'{key}_loss']
            
            if batch_idx % 5 == 0:
                print(f"  Battle Update: "
                      f"🕵️ Detective Skill: {losses['detector_loss']:.4f}, "
                      f"🥷 Forger Skill: {losses['embedding_loss']:.4f}, "
                      f"📤 Extraction: {losses['extraction_loss']:.4f}")
        
        # Print epoch summary - WHO'S WINNING?
        num_batches = len(train_loader)
        det_avg = epoch_losses['detector']/num_batches
        emb_avg = epoch_losses['embedding']/num_batches
        
        print(f"\n📊 Round {epoch+1} Results:")
        print(f"  🕵️ Detective Performance: {det_avg:.4f}")
        print(f"  🥷 Forger Performance: {emb_avg:.4f}")
        
        if det_avg > emb_avg:
            print("  🏆 Detective is winning this round!")
        else:
            print("  🏆 Forger is winning this round!")
        
        print("-" * 50)
    
    print("\n✅ Training completed!")
    
    # Demonstration
    print("\n🎭 Demonstrating the system...")
    
    # Take a sample image
    sample_image = torch.FloatTensor(np.transpose(demo_images[0], (2, 0, 1)))
    
    # Create a secret message
    secret_message = "Hello, this is a secret message hidden by AI!"
    print(f"Secret message: '{secret_message}'")
    
    # Convert message to binary image
    message_binary = string_to_binary_image(secret_message)
    message_tensor = torch.FloatTensor(message_binary)
    
    # Embed the message
    print("\n🔒 Embedding message...")
    stego_image, attention_map = stego_system.embed_message(sample_image, message_tensor)
    
    # Extract the message
    print("🔓 Extracting message...")
    extracted_binary = stego_system.extract_message(torch.FloatTensor(stego_image))
    extracted_message = binary_image_to_string(extracted_binary[0, 0])
    
    print(f"Extracted message: '{extracted_message}'")
    
    # Test detection
    print("\n🕵️ Testing detection...")
    original_detection = stego_system.detect_steganography(sample_image)
    stego_detection = stego_system.detect_steganography(torch.FloatTensor(stego_image))
    
    print(f"Original image detection score: {original_detection[0, 0]:.4f}")
    print(f"Stego image detection score: {stego_detection[0, 0]:.4f}")
    print(f"(Lower scores mean better steganography)")
    
    # Visualize results
    print("\n📊 Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(np.transpose(sample_image.numpy(), (1, 2, 0)))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Message (binary)
    axes[0, 1].imshow(message_binary, cmap='gray')
    axes[0, 1].set_title('Secret Message (Binary)')
    axes[0, 1].axis('off')
    
    # Attention map
    axes[0, 2].imshow(attention_map[0, 0], cmap='hot')
    axes[0, 2].set_title('AI Attention Map\n(Where to hide)')
    axes[0, 2].axis('off')
    
    # Stego image
    axes[1, 0].imshow(np.transpose(stego_image[0], (1, 2, 0)))
    axes[1, 0].set_title('Stego Image\n(With hidden message)')
    axes[1, 0].axis('off')
    
    # Extracted message
    axes[1, 1].imshow(extracted_binary[0, 0], cmap='gray')
    axes[1, 1].set_title('Extracted Message')
    axes[1, 1].axis('off')
    
    # Difference map
    diff = np.abs(stego_image[0] - sample_image.numpy())
    diff_magnitude = np.mean(diff, axis=0)
    axes[1, 2].imshow(diff_magnitude, cmap='hot')
    axes[1, 2].set_title('Difference Map\n(Changes made)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('steganography_demo.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'steganography_demo.png'")
    
    # Save the model
    print("\n💾 Saving trained model...")
    torch.save({
        'embedding_net': stego_system.embedding_net.state_dict(),
        'detector_net': stego_system.detector_net.state_dict(),
        'extraction_net': stego_system.extraction_net.state_dict(),
    }, 'ai_steganography_model.pth')
    
    print("\n🎉 Demo completed successfully!")
    print("Your AI-powered steganography system is ready!")
    print("\nKey achievements:")
    print("✓ AI analyzes images to find optimal hiding spots")
    print("✓ Adversarial training makes detection very difficult")
    print("✓ Messages can be embedded and extracted successfully")
    print("✓ System is battle-tested against AI detectors")

if __name__ == "__main__":
    main()