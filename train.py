import argparse
import data_utils
import network_utils
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Train Deep Learning Model for Flower Classification")
    parser.add_argument('data_dir', type=str, help="data directory(required)")
    parser.add_argument('--save_dir', default='', type=str, help="directory to save checkpoints")
    parser.add_argument('--arch', default='resnet18',
                        help='Deep NN architecture, options: resnet18, resnet34, resnet50, resnet101')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_units', default=256, type=int, help='number of neurons in hidden layer')
    parser.add_argument('--output_units', default=102, type=int, help='number of output categories')
    parser.add_argument('--drop_prob', default=0.1, type=float, help='dropout probability')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs for training')
    parser.add_argument('--gpu', default=False, action='store_true', help='GPU to be used for training?')
    return parser.parse_args()


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, print_every, use_gpu):
    if use_gpu and torch.cuda.is_available():
        print("Using GPU")
        model.cuda()

    steps = 0
    for e in range(epochs):
        running_loss = 0
        print(f"Epoch {e + 1} -------------------------------------------")
        for images, labels in train_loader:
            steps += 1
            if use_gpu and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                t_loss = running_loss / print_every
                v_loss, v_acc = validate(model, valid_loader, criterion, use_gpu)
                print(f"Training Loss: {t_loss:.4f} Validation Loss: {v_loss:.4f} Validation Acc: {v_acc:.4f}")
                running_loss = 0
                model.train()


def validate(model, data_loader, criterion, use_gpu):
    if use_gpu and torch.cuda.is_available():
        model.to('cuda')

    loss = 0
    acc = 0

    for images, labels in data_loader:
        if use_gpu and torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')

        with torch.no_grad():
            outputs = model.forward(images)
            loss += criterion(outputs, labels)
            preds = torch.exp(outputs).data

            equality = (labels.data == preds.max(1)[1])
            acc += equality.type_as(torch.FloatTensor()).mean()

    loss /= len(data_loader)
    acc /= len(data_loader)

    return loss, acc


def main():
    args = get_args()
    print_training_config(args)
    train_loader, valid_loader, test_loader, class_to_idx = data_utils.get_data_loaders(args.data_dir)
    model = network_utils.build_network(args.arch, args.hidden_units, args.output_units, args.drop_prob)
    model.class_to_idx = class_to_idx
    criterion = network_utils.get_loss_function()
    optimizer = network_utils.get_optimizer(model, args.learning_rate)
    train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, 10, args.gpu)
    network_utils.save_model(model, args.save_dir, args.arch, args.epochs, args.learning_rate, args.hidden_units)


def print_training_config(args):
    print("Training Configuration:")
    print(f"Architecture: {args.arch}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Hidden Units: {args.hidden_units}")
    print(f"Output Units: {args.output_units}")
    print(f"Dropout Probability: {args.drop_prob}")
    print(f"Epochs: {args.epochs}")
    print(f"Use GPU?: {args.gpu}")


if __name__ == '__main__':
    main()
