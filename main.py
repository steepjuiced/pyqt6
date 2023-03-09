import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QPixmap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import Isomap


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # create labels and buttons
        label1 = QLabel('Select Image 1:')
        self.btn1 = QPushButton('Browse', self)
        label2 = QLabel('Select Image 2:')
        self.btn2 = QPushButton('Browse', self)
        self.run_button = QPushButton('Run', self)

        # connect file browser buttons to browse_file method
        self.btn1.clicked.connect(lambda: self.browse_file(1))
        self.btn2.clicked.connect(lambda: self.browse_file(2))
        self.run_button.clicked.connect(self.run)

        # create vertical box layout and add widgets
        vbox = QVBoxLayout()
        vbox.addWidget(label1)
        vbox.addWidget(self.btn1)
        vbox.addWidget(label2)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.run_button)

        # set layout and window properties
        self.setLayout(vbox)
        self.setWindowTitle('My App')
        self.setGeometry(100, 100, 400, 300)
        self.show()

    def browse_file(self, img_num):
        # open file browser and set selected file path in corresponding button text
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Images (*.png *.xpm *.jpg)')
        if img_num == 1:
            self.btn1.setText(filename)
            self.img1 = plt.imread(filename)
        elif img_num == 2:
            self.btn2.setText(filename)
            self.img2 = plt.imread(filename)

    def run(self):
        # compute L2 distance between images and plot them
        dist = euclidean_distances(self.img1.reshape(1, -1), self.img2.reshape(1, -1))[0, 0]
        fig, ax = plt.subplots()
        ax.imshow(np.concatenate((self.img1, np.ones((self.img1.shape[0], 10, 3)), self.img2), axis=1))
        ax.set_title(f"L2 distance: {dist:.2f}")
        plt.show()

        # perform Isomap embedding and plot result
        X = np.vstack((self.img1.reshape(1, -1), self.img2.reshape(1, -1)))
        iso = Isomap(n_components=2)
        X_iso = iso.fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(X_iso[:, 0], X_iso[:, 1])
        ax.set_title('Isomap Embedding')
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec())
