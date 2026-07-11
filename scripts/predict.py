import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.datasets.stl10_labelled_dataset import STL10LabelledDataset
from src.models.classifier import STL10Classifier

def main(config_path, index):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    assert config['image_size'] % config['patch_size'] == 0, \
        "image_size must be divisible by patch_size"
    config['grid_size'] = config['image_size'] // config['patch_size']
    config['max_seq_len'] = config['grid_size'] ** 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = STL10LabelledDataset(config['test_data_path'], config['test_labels_path'])

    encoder_kwargs = dict(
        in_channels=config['in_channels'], patch_size=config['patch_size'],
        d_model=config['encoder_d_model'], d_ff=config['encoder_d_ff'], num_heads=config['encoder_num_heads'],
        num_layers=config['encoder_num_layers'], max_seq_len=config['max_seq_len'],
    )  

    model = STL10Classifier(encoder_kwargs, num_classes=config['num_classes']).to(device)

    checkpoint = torch.load(config['checkpoint_dir'] + "/model_step_final.pt", map_location='cpu', weights_only=False)

    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)

    img = test_dataset[index][0]
    actual_label = test_dataset[index][1]

    input = img.permute(2, 0, 1).unsqueeze(0).to(device)  # Add batch dimension and move to device

    predicted_label = model.predict(input)

    plt.imshow(img)

    plt.title(f"Actual: {actual_label}, Predicted: {predicted_label.item()}")
    plt.axis('off')
    plt.savefig(f"prediction.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict using I-JEPA on STL-10 Labelled Dataset')
    parser.add_argument('--config', type=str, default='configs/stl10_ijepa_classifier.yaml', help='Path to the training configuration YAML file')
    parser.add_argument('index', type=int, help='Index of the test sample to predict')

    args = parser.parse_args()

    main(args.config, args.index)