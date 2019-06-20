from argparse import ArgumentParser
from classify.cnn_model import CNNModel
from classify.logistic_model import LogisticModel
from classify.fisher_model import FisherModel
from classify.svm_model import SVMModel
from detect.detect import Detector
import cv2
from preprocess.data_loader import data_loader


def main(args):
    model = None
    if args.model == 'fisher':
        model = FisherModel()
    elif args.model == 'cnn':
        model = CNNModel()
    elif args.model == 'logistic':
        model = LogisticModel()
    elif args.model == 'svm':
        model = SVMModel()
    
    assert model is not None, f'model {args.model} not available'
    
    if args.function == 'classify':
        if args.mode == 'train':
            from_h5 = False
            if args.model == 'cnn':
                from_h5 = True
            x_training, y_training = data_loader(mode='train', from_h5=from_h5)
            x_testing, y_testing = data_loader(mode='test', from_h5=from_h5)
            model.load_data(x_training, y_training, x_testing, y_testing)

            if args.model == 'logistic':
                model.train(solver=args.solver)
            elif args.model == 'fisher':
                model.train()
            elif args.model == 'cnn':
                model.train()
            elif args.model == 'svm':
                model.train(kernel=args.kernel)
            
            model.save(args.save_path)
        
        if args.mode == 'test':
            model.load(args.load_path)
            img = cv2.imread(args.image)
            p = model.predict([img])[0]
            img = cv2.putText(img, str(round(p, 3)), (img.shape[1]//2-20, img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey()
    
    if args.function == 'detect':
        model.load(args.load_path)
        print(model)
        detector = Detector(model)
        img = cv2.imread(args.image)
        result = detector.detect(img)
        print(result.size)
        if args.save_image:
            cv2.imwrite(args.save_image, result)
        
        cv2.imshow('result', result)
        cv2.waitKey()



if __name__ == '__main__':
    parser = ArgumentParser()

    parser_func = parser.add_subparsers(dest='function')
    parser_func.required = True

    parser_classify = parser_func.add_parser('classify')

    subparser_classify = parser_classify.add_subparsers(dest='mode')
    subparser_classify.required = True

    parser_classify_train = subparser_classify.add_parser('train')
    parser_classify_train.add_argument('--model', required=True)
    parser_classify_train.add_argument('--save-path', required=True)
    parser_classify_train.add_argument('--solver')
    parser_classify_train.add_argument('--kernel')

    parser_classify_test = subparser_classify.add_parser('test')
    parser_classify_test.add_argument('--model', required=True)
    parser_classify_test.add_argument('--load-path', required=True)
    parser_classify_test.add_argument('--image', required=True)

    parser_detect = parser_func.add_parser('detect')
    parser_detect.add_argument('--model', required=True)
    parser_detect.add_argument('--load-path', required=True)
    parser_detect.add_argument('--image', required=True)
    parser_detect.add_argument('--save-image')

    print(parser.parse_args())
    main(parser.parse_args())