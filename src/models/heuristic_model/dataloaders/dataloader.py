from torch.utils.data import DataLoader
from .dataset import VQA_X_Dataset, load_data, build_vocabularies

def get_dataloaders(config):
    """
    Create DataLoader objects for training, validation, and testing.
    
    Args:
        config (dict): Configuration dictionary containing data paths and parameters.
        
    Returns:
        tuple: Train, validation, and test dataloaders, along with vocabulary mappings.
    """
    # Load datasets
    train_data = load_data(config['data']['train_path'])
    val_data = load_data(config['data']['val_path'])
    test_data = load_data(config['data']['test_path'])

    # Build vocabularies from training data
    word2idx, idx2word, answer2idx, idx2answer = build_vocabularies(train_data)

    # Create dataset instances
    train_dataset = VQA_X_Dataset(
        train_data,
        config['data']['train_image_dir'],
        word2idx=word2idx,
        idx2word=idx2word,
        answer2idx=answer2idx,
        idx2answer=idx2answer
    )

    val_dataset = VQA_X_Dataset(
        val_data,
        config['data']['val_image_dir'],
        word2idx=word2idx,
        idx2word=idx2word,
        answer2idx=answer2idx,
        idx2answer=idx2answer
    )

    test_dataset = VQA_X_Dataset(
        test_data,
        config['data']['test_image_dir'],
        word2idx=word2idx,
        idx2word=idx2word,
        answer2idx=answer2idx,
        idx2answer=idx2answer
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers'],
        pin_memory=True
    )

    return (
        train_loader, 
        val_loader, 
        test_loader, 
        word2idx, 
        idx2word, 
        answer2idx, 
        idx2answer
    )