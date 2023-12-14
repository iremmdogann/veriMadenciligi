import numpy as np
import matplotlib.pyplot as plt

def merkezi_hesapla(X, pencere_boyutu):
    merkezler = []
    for x in X:
        pencere_icindeki_noktalar = X[np.linalg.norm(X - x, axis=1) < pencere_boyutu]
        yeni_merkez = np.mean(pencere_icindeki_noktalar, axis=0)
        merkezler.append(yeni_merkez)
    return np.array(merkezler)

def mean_shift(X, pencere_boyutu, maks_iterasyon=100):
    for _ in range(maks_iterasyon):
        yeni_merkezler = merkezi_hesapla(X, pencere_boyutu)
        hata = np.linalg.norm(yeni_merkezler - X, axis=1)
        X = yeni_merkezler

        if np.all(hata < 1e-5):
            break

    etiketler = np.arange(len(X))
    return X, etiketler

# Örnek veri oluştur
np.random.seed(42)
X = np.concatenate([np.random.normal(0, 1, (50, 2)), np.random.normal(5, 1, (50, 2))])

# Mean Shift algoritması ile kümeleme
pencere_boyutu = 1.5
sonuclar, etiketler = mean_shift(X, pencere_boyutu)

plt.scatter(X[:, 0], X[:, 1], c=etiketler, cmap='viridis', s=50, alpha=0.7)
plt.scatter(sonuclar[:, 0], sonuclar[:, 1], c='red', marker='X', s=200, label='Merkezler')
plt.title('Mean Shift Kümeleme')
plt.legend()
plt.show()
