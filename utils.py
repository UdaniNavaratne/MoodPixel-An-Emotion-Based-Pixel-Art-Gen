from models import ResNet50, LSTMPyTorch
import torch

def load_models():
    print("🔍 Loading ResNet50 model...")
    backbone = ResNet50(num_classes=7, channels=3)
    
    state_dict = torch.load("models/FER_static_ResNet50_AffectNet.pt", map_location='cpu')
    print("✅ Keys in loaded state_dict:")
    print(list(state_dict.keys())[:10])  # print only first 10 keys
    
    backbone.load_state_dict(state_dict)
    backbone.eval()

    print("✅ ResNet50 loaded.")

    print("🔍 Loading LSTM model...")
    lstm = LSTMPyTorch()
    lstm.load_state_dict(torch.load("models/FER_dinamic_LSTM_Aff-Wild2.pt", map_location='cpu'))
    lstm.eval()
    print("✅ LSTM loaded.")

    return backbone, lstm
