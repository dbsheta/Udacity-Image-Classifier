import argparse
import image_utils
import network_utils
import torch
import json


def get_args():
    parser = argparse.ArgumentParser(description="Train Deep Learning Model for Flower Classification")
    parser.add_argument('input', type=str, help="input image(required)")
    parser.add_argument('checkpoint', type=str, help='pre-trained model path')
    parser.add_argument('--top_k', default=3, type=int, help='default top_k results')
    parser.add_argument('--category_names', default='', type=str, help='default category file')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU?')
    return parser.parse_args()


def predict(image, model, topk, category_names, use_gpu):
    model.eval()
    class_to_idx = model.class_to_idx
    idx_to_class = {class_to_idx[k]: k for k in class_to_idx}
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image, dim=0)

    if use_gpu and torch.cuda.is_available():
        print("Using GPU")
        image.cuda()
        model.cuda()

    with torch.no_grad():
        output = model.forward(image)
        preds = torch.exp(output).topk(topk)
    probs = preds[0][0].cpu().data.numpy().tolist()
    classes = preds[1][0].cpu().data.numpy()
    classes = [idx_to_class[i] for i in classes]
    if category_names != '':
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[x] for x in classes]

    return probs, classes


def main():
    args = get_args()
    processed_img = image_utils.process_image(args.input)
    model = network_utils.load_model(args.checkpoint)
    probs, classes = predict(processed_img, model, args.top_k, args.category_names, args.gpu)
    print(f"Top {args.top_k} predictions: {list(zip(classes, probs))}")


if __name__ == '__main__':
    main()
