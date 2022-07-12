import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('pointnet2_classification_stats.pkl', 'rb') as fh:
        data_pn2 = pickle.load(fh)
    with open('point_transformer_classification_stats.pkl', 'rb') as fh:
        data_pn_transf = pickle.load(fh)

    acc_1 = data_pn2['test_acc']
    acc_2 = data_pn_transf['test_acc']

    plt.figure(figsize=(9,7))
    plt.plot(acc_1, label='PointNet++')
    plt.plot(acc_2, label='Point Transformer')
    plt.title('Test Accuracy, ModelNet10 Classification', size=20)
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower left')
    plt.savefig('comparison.png')