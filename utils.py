from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_transforms(kind: str):
    if kind == 'train':
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif kind in ['val', 'test']:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        raise ValueError("kind must be one of 'train', 'val' or 'test'.")


def load_img_data(dir, kind: str = 'train'):
    return datasets.ImageFolder(dir, transform=get_transforms(kind))


def plot_prediction(top_probs, top_classes, img, img_name: str = ''):
    _, axes = plt.subplots(1, 2, layout='tight')
    
    # Show image
    axes[0].axis('off')
    axes[0].set_title(img_name)
    axes[0].imshow(img)

    # Show classification results
    axes[1].set_title('Top Class Probability')
    axes[1].barh(top_classes, top_probs)
    axes[1].set_xlim(0, 1.1)

    plt.show()
