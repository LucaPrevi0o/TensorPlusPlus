# Tensor++ - Libreria C++ per tensori multi-dimensionali

## Descrizione

Tensor++ è una libreria C++ che fornisce una classe template generica per la gestione di tensori (array multi-dimensionali) di qualsiasi tipo e dimensione.
Include operatori matematici, funzioni di utilità per matrici e tuple, e supporto per operazioni avanzate come contrazione di tensori e calcolo del determinante.

## Funzionalità principali
- Classe `tensor<A, N>`: tensore generico di tipo `A` e dimensione `N`
- Alias `matrix<A>` e `tuple<A>` per matrici 2D e tuple 1D
- Operatori matematici: somma, sottrazione, moltiplicazione tra tensori e scalari
- Funzioni di utilità: trasposizione, traccia, submatrice, adjugata, determinante
- Controllo automatico degli indici e delle dimensioni

## Utilizzo

### Inclusione
La libreria può essere utilizzata come semplice header:
```cpp
#include "tensor.h"
```

### Creazione di un tensore
La creazione di un nuovo tensore richiede un numero di parametri pari alla dimensione del suo spazio vettoriale, che ne identificano il numero di elementi per ogni direzione:
```cpp
tensor<double, 3> t(2, 3, 4); // Tensore 2x3x4 di double
```
E' inoltre possibile utilizzare direttamente gli alias `tuple` e `matrix` per la definizione di tensore 1D e 2D:
```cpp
matrix<double> m(3, 3); // Matrice 3x3
tuple<int> v(5);        // Tupla di 5 elementi
```

### Operazioni
```cpp
auto t2 = t + t;   // Somma tra tensori
auto t3 = t + 5.0; // Somma con scalare
auto t4 = t * 2.0; // Moltiplicazione con scalare
```

### Funzioni di utilità
```cpp
auto mt    = tensor::T(m);   // Matrice trasposta
auto trace = tensor::tr(m);  // Traccia della matrice
auto det   = tensor::det(m); // Determinante della matrice
auto adj   = tensor::adj(m); // Matrice aggiunta

// Righe e colonne della sottomatrice
auto row   = tensor::tuple<int>(1); // una riga da rimuovere
auto col   = tensor::tuple<int>(2); // due colonne da rimuovere

row(0) = 1; // Riga 1 della matrice da rimuovere
col(0) = 3; // Colonna 3 della matrice da rimuovere
col(1) = 4; // Colonna 4 della matrice da rimuovere
auto subm  = tensor::submatrix(m, row, col); // Sottomatrice rispetto alle righe/colonne
```

## API Principali

### Classe `tensor<A, N>`
- Costruttori: dimensioni variabili, copia, assegnazione
- Operatori: somma, sottrazione, moltiplicazione
- Metodi statici: `tensor<A, N> zero(...)` (tensore nullo), `tensor<A, N> identity(...)` (tensore identità)

### Funzioni globali
Operatori vettoriali:
- `tuple<A> sort(tuple<A>, bool: true)` - ordinamento (default: ascendente)
- `tuple<A> reverse(tuple<A>)` - inversione
- `A dot(tuple<A>, tuple<A>)` - prodotto scalare
- `A norm(tuple<A>)` - norma euclidea

Operatori matriciali:
- `matrix<A> dot(matrix<A>, matrix<A>)` - prodotto scalare (elemento per elemento)
- `matrix<A> T(matrix<A>)` — trasposizione
- `A tr(matrix<A>)` — traccia
- `matrix<A> submatrix(matrix<A>, tuple<int>, tuple<int>)` — sottomatrice
- `matrix<A> adj(matrix<A>)` — matrice aggiunta
- `A det(matrix<A>)` — determinante

## Esempio completo
```cpp
#include "tensor.h"
using namespace tensor;

int main() {

    matrix<double> m(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m(i, j) = i + j;
    auto mt = T(m);
    auto trace = tr(m);
    auto detm = det(m);
}
```

## Autore
@[LucaPrevi0o](https://github.com/LucaPrevi0o) - Luca Previati

## Licenza
MIT
