#ifndef TENSOR_H
#define TENSOR_H

namespace tensor {

    /**
     * @brief Tensor class for multi-dimensional arrays.
     * 
     * This class represents a tensor, which is a multi-dimensional array.
     * It provides methods for accessing and manipulating the tensor data.
     * * @tparam A Type of the tensor elements
     * * @tparam N Number of dimensions of the tensor
     */
    template<typename A, int N>
    class tensor {

        private:

            A* data; // Pointer to the tensor data
            int capacity[N]; // Array to hold the size of each dimension
            int strides[N]; // Array to hold the stride for each dimension (pre-calculated)

            /**
             * @brief Construct a new tensor object
             * 
             * @param dims Array of sizes for each dimension
             */
            tensor(int dims[N]) {

                for (int i = 0; i < N; i++) {
                    
                    capacity[i] = dims[i];
                    if (dims[i] <= 0) throw "Invalid tensor size";
                }
                calculate_strides();
                data = new A[length()];
            }

            /**
             * @brief Calculate the total number of elements in the tensor.
             * 
             * @return Total number of elements in the tensor
             */
            int length() const {
                
                int size = 1;
                for (int i = 0; i < N; i++) size *= capacity[i]; // Calculate the total size of the tensor
                return size; // Return the total size
            }

            /**
             * @brief Check if the indices of another tensor match this tensor's dimensions.
             * 
             * @param other The tensor to compare with
             * @throw "Index does not match tensor dimensions" if the indices do not match
             */
            void check_indices(const tensor& other) const {

                for (int i = 0; i < N; i++)
                    if (other.capacity[i] != capacity[i]) throw "Index does not match tensor dimensions";
            }

            /**
             * @brief Calculate and store strides for this tensor.
             * 
             * Strides represent the number of elements to skip in the linear array
             * to move one position in each dimension.
             */
            void calculate_strides() {

                strides[N - 1] = 1;
                for (int i = N - 2; i >= 0; --i) strides[i] = strides[i + 1] * capacity[i + 1];
            }

            /**
             * @brief Determine the dimension to be broadcasted between two tensors.
             * 
             * Broadcasting is allowed if all dimensions match except one which is 1.
             * 
             * @param other The tensor to compare with
             * @return The index of the dimension to be broadcasted, or -1 if no broadcasting is needed
             * @throw "Index does not match tensor dimensions (broadcasting not valid)" if broadcasting is not possible
             */
            int broadcasting_dim(const tensor& other) const {

                int broadcast_dim = -1; // Dimension to be broadcasted, if any
                for (int i = 0; i < N; ++i) if (capacity[i] != other.capacity[i]) {

                    if (capacity[i] == 1 && other.capacity[i] > 1 && broadcast_dim == -1) broadcast_dim = i; // broadcast this tensor
                    else if (other.capacity[i] == 1 && capacity[i] > 1 && broadcast_dim == -1) broadcast_dim = i; // broadcast other tensor
                    else throw "Index does not match tensor dimensions (broadcasting not valid)"; // incompatible sizes
                }
                return broadcast_dim;
            }

            /**
             * @brief Helper function to sum two tensors with broadcasting.
             * 
             * This function is used internally to perform element-wise addition
             * between two tensors, taking into account broadcasting rules.
             * 
             * @param t The tensor to add to this tensor
             * @param other The tensor to be added
             * @param idx The linear index of the element to be summed
             * @return The result of the summation for the specified index
             */
            static A sum_value(const tensor& t, const tensor& other, int idx) {

                // Extract multi-dimensional indices using pre-calculated strides
                int indices[N];
                int tmp = idx;
                for (int d = 0; d < N; ++d) {

                    indices[d] = tmp / t.strides[d];
                    tmp = tmp % t.strides[d];
                }
                
                // Calculate index in the other tensor using its pre-calculated strides
                int other_idx = 0;
                for (int d = 0; d < N; ++d) {

                    int ind = (other.capacity[d] == 1) ? 0 : indices[d];
                    other_idx += ind * other.strides[d];
                }
                return t.data[idx] + other.data[other_idx];
            }

            /**
             * @brief Helper function to subtract two tensors with broadcasting.
             * 
             * This function is used internally to perform element-wise subtraction
             * between two tensors, taking into account broadcasting rules.
             * 
             * @param t The tensor from which to subtract
             * @param other The tensor to be subtracted
             * @param idx The linear index of the element to be subtracted
             * @return The result of the subtraction for the specified index
             */
            static A sub_value(const tensor& t, const tensor& other, int idx) {

                // Extract multi-dimensional indices using pre-calculated strides
                int indices[N];
                int tmp = idx;
                for (int d = 0; d < N; ++d) {

                    indices[d] = tmp / t.strides[d];
                    tmp = tmp % t.strides[d];
                }
                
                // Calculate index in the other tensor using its pre-calculated strides
                int other_idx = 0;
                for (int d = 0; d < N; ++d) {

                    int ind = (other.capacity[d] == 1) ? 0 : indices[d];
                    other_idx += ind * other.strides[d];
                }
                return t.data[idx] - other.data[other_idx];
            }

        public:

            /**
             * @brief Destructor for the tensor class.
             * 
             * Frees the allocated memory for the tensor data.
             */
            ~tensor() { delete[] data; } // Destructor to free allocated memory

            /**
             * @brief Constructor for the tensor class with variable number of arguments.
             * 
             * Initializes the tensor with specified dimensions using variadic template.
             * 
             * @param args Variable number of size arguments (must match tensor dimensions N)
             */
            template<typename... Args>
            tensor(Args... args) {

                if (sizeof...(args) != N) throw "Number of size arguments must match tensor dimensions";
                int dims[] = {args...};
                for (int i = 0; i < N; ++i) capacity[i] = dims[i];
                calculate_strides(); // Pre-calculate strides for efficient indexing
                data = new A[length()]; // Allocate memory for the tensor data
            }

            /**
             * @brief Copy constructor for the tensor class.
             * 
             * Initializes a new tensor as a copy of another tensor.
             * 
             * @param other The tensor to copy from
             */
            tensor(const tensor& other) {

                for (int i = 0; i < N; ++i) {
                    capacity[i] = other.capacity[i];
                    strides[i] = other.strides[i];
                }
                data = new A[length()];
                for (int i = 0; i < length(); ++i) data[i] = other.data[i];
            }

            /**
             * @brief Return the size of the tensor in each dimension.
             * 
             * @return Array of sizes for each dimension
             */
            const int* size() const { return capacity; }

            /**
             * @brief Return the size of a specific dimension.
             * @param index Dimension index 
             * @return Size of the specified dimension
             */
            int size(int index) const { return capacity[index]; }

            /**
             * @brief Access the element at the specified index using variadic arguments.
             * 
             * @param args Variable number of indices (must match tensor dimensions N)
             * @return Reference to the element at the specified index
             */
            template<typename... Args>
            A& operator()(Args... args) const {

                if (sizeof...(args) != N) throw "Number of indices must match tensor dimensions";
                int indices[] = {args...}; // Pack arguments into array
                
                int index = 0;
                for (int i = 0; i < N; i++) {
                    
                    if (indices[i] < 0 || indices[i] >= capacity[i]) throw "Index out of bounds";
                    index = index * capacity[i] + indices[i];
                }
                return data[index];
            }

            /**
             * @brief Add two tensors element-wise.
             * 
             * @param other The tensor to add to this tensor
             * @return A new tensor containing the result of the addition
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator+(const tensor& other) const {

                int broadcast_dim = broadcasting_dim(other); // Determine the dimension to be broadcasted, if any
                tensor result(*this); // Create a new tensor to hold the result

                if (broadcast_dim == -1) for (int i = 0; i < length(); i++) result.data[i] = data[i] + other.data[i];
                else for (int idx = 0; idx < length(); ++idx) result.data[idx] = tensor::sum_value(*this, other, idx);
                return result;
            }

            /**
             * @brief Add a scalar to each element of the tensor.
             * 
             * @param scalar The scalar to add
             * @return A new tensor containing the result of the addition
             */
            tensor operator+(const A& scalar) const {

                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] + scalar; // Add the scalar to each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Add a scalar to each element of the tensor.
             * 
             * @param scalar The scalar to add
             * @param t The tensor to which the scalar is added
             * @return A new tensor containing the result of the addition
             */
            friend tensor operator+(const A& scalar, const tensor& t) {

                tensor result(t); // Create a new tensor to hold the result
                for (int i = 0; i < t.length(); i++) result.data[i] = scalar + t.data[i]; // Add the scalar to each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Add another tensor to this tensor element-wise.
             * 
             * @param other The tensor to add to this tensor
             * @return Reference to the current tensor after addition
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator+=(const tensor& other) {

                int broadcast_dim = broadcasting_dim(other); // Determine the dimension to be broadcasted, if any

                if (broadcast_dim == -1) for (int i = 0; i < length(); i++) data[i] += other.data[i];
                else for (int idx = 0; idx < length(); ++idx) data[idx] = tensor::sum_value(*this, other, idx);
                return *this;
            }

            /**
             * @brief Add a scalar to each element of this tensor.
             * 
             * @param scalar The scalar to add
             * @return Reference to the current tensor after addition
             */
            tensor operator+=(const A& scalar) {

                for (int i = 0; i < length(); i++) data[i] += scalar; // Add the scalar to each element of the tensor
                return *this; // Return the current tensor
            }

            /**
             * @brief Subtract two tensors element-wise.
             * 
             * @param other The tensor to subtract from this tensor
             * @return A new tensor containing the result of the subtraction
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator-(const tensor& other) const {

                // Broadcasting: consenti sottrazione se tutte le dimensioni coincidono tranne una che è 1
                int broadcast_dim = broadcasting_dim(other);

                tensor result(*this);
                if (broadcast_dim == -1) for (int i = 0; i < length(); i++) result.data[i] = data[i] - other.data[i];
                else for (int idx = 0; idx < length(); ++idx) result.data[idx] = tensor::sub_value(*this, other, idx);
                return result;
            }

            /**
             * @brief Subtract a scalar from each element of the tensor.
             * 
             * @param scalar The scalar to subtract
             * @return A new tensor containing the result of the subtraction
             */
            tensor operator-(const A& scalar) const {

                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] - scalar; // Subtract the scalar from each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Subtract a scalar from each element of the tensor.
             * 
             * @param scalar The scalar to subtract
             * @param t The tensor from which the scalar is subtracted
             * @return A new tensor containing the result of the subtraction
             */
            friend tensor operator-(const A& scalar, const tensor& t) {

                tensor result(t); // Create a new tensor to hold the result
                for (int i = 0; i < t.length(); i++) result.data[i] = scalar - t.data[i]; // Subtract the scalar from each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Subtract another tensor from this tensor element-wise.
             * 
             * @param other The tensor to subtract from this tensor
             * @return Reference to the current tensor after subtraction
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator-=(const tensor& other) {

                // Broadcasting: consenti sottrazione se tutte le dimensioni coincidono tranne una che è 1
                int broadcast_dim = broadcasting_dim(other);

                if (broadcast_dim == -1) for (int i = 0; i < length(); i++) data[i] -= other.data[i];
                else for (int idx = 0; idx < length(); ++idx) data[idx] = tensor::sub_value(*this, other, idx);
                return *this;
            }
            
            /**
             * @brief Multiply two tensors using tensor contraction.
             * 
             * This method performs tensor contraction between two tensors.
             * The last dimension of the first tensor must match the first dimension of the second tensor.
             * 
             * @param other The tensor to multiply with
             * @return A new tensor containing the result of the contraction
             * @throw "Tensor contraction: dimensions do not match" if the last dimension of the first tensor does not match the first dimension of the second tensor
             * * @tparam M Number of dimensions of the second tensor
             */ 
            template<int M> tensor<A, N + M - 2> operator*(const tensor<A, M>& other) const {

                // Contrai sull'ultima dimensione di this e la prima di other
                if (capacity[N - 1] != other.capacity[0]) throw "Tensor contraction: dimensions do not match";

                // Calcola le dimensioni del tensore risultato
                int new_dims[N + M - 2], idx = 0;
                for (int i = 0; i < N - 1; ++i) new_dims[idx++] = capacity[i];
                for (int i = 1; i < M; ++i) new_dims[idx++] = other.capacity[i];

                tensor<A, N + M - 2> result(new_dims);

                // Calcola il numero di elementi da contrarre
                int contracted = capacity[N - 1];

                // Calcola il numero di elementi nelle parti non contratte
                int left_size = 1;
                for (int i = 0; i < N - 1; ++i) left_size *= capacity[i];
                int right_size = 1;
                for (int i = 1; i < M; ++i) right_size *= other.capacity[i];

                for (int i = 0; i < left_size; ++i)
                    for (int j = 0; j < right_size; ++j) {

                        A sum = 0;
                        for (int k = 0; k < contracted; ++k) sum += data[i * contracted + k] * other.data[k * right_size + j];
                        result.data[i * right_size + j] = sum;
                    }

                return result;
            }

            /**
             * @brief Multiply a scalar by each element of the tensor.
             * 
             * @param scalar The scalar to multiply by
             * @return A new tensor containing the result of the multiplication
             */
            tensor operator*(const A& scalar) const {

                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] * scalar; // Multiply each element of the tensor by the scalar
                return result; // Return the resulting tensor
            }

            /**
             * @brief Multiply a scalar by each element of the tensor.
             * 
             * @param scalar The scalar to multiply by
             * @param t The tensor to which the scalar is multiplied
             * @return A new tensor containing the result of the multiplication
             */
            friend tensor operator*(const A& scalar, const tensor& t) {

                tensor result(t); // Create a new tensor to hold the result
                for (int i = 0; i < t.length(); i++) result.data[i] = scalar * t.data[i]; // Multiply each element of the tensor by the scalar
                return result; // Return the resulting tensor
            }

            tensor operator*=(const A& scalar) {

                for (int i = 0; i < length(); i++) data[i] *= scalar; // Multiply each element of the tensor by the scalar
                return *this; // Return the current tensor
            }

            /**
             * @brief Check if two tensors are equal.
             * 
             * Compares the sizes of the dimensions and the elements of the tensors.
             * 
             * @param other The tensor to compare with
             * @return true if the tensors are equal, false otherwise
             */
            bool operator==(const tensor& other) const {

                for (int i = 0; i < N; i++)
                    if (capacity[i] != other.capacity[i]) return false; // Check if the sizes of the dimensions match
                for (int i = 0; i < length(); i++)
                    if (data[i] != other.data[i]) return false; // Check if the elements are equal
                return true; // Tensors are equal
            }

            /**
             * @brief Check if two tensors are not equal.
             * 
             * Compares the sizes of the dimensions and the elements of the tensors.
             * 
             * @param other The tensor to compare with
             * @return true if the tensors are not equal, false otherwise
             */
            bool operator!=(const tensor& other) const { return !(*this == other); }

            /**
             * @brief Assign the value of another tensor to this tensor.
             * 
             * Copies the sizes of the dimensions and the elements from another tensor.
             * 
             * @param other The tensor to copy from
             * @return Reference to the current tensor after assignment
             */
            tensor& operator=(const tensor& other) {

                if (this == &other) return *this; // Check for self-assignment

                delete[] data; // Delete the old data
                for (int i = 0; i < N; i++) {
                    
                    capacity[i] = other.capacity[i]; // Copy the sizes of the dimensions
                    strides[i] = other.strides[i]; // Copy the strides
                }
                data = new A[length()]; // Allocate new memory for the tensor data
                for (int i = 0; i < length(); i++) data[i] = other.data[i]; // Copy the elements of the tensor
                return *this; // Return the current tensor
            }

            /**
             * @brief Assign the value of an array to this tensor.
             * 
             * Copies the elements from an array to the tensor.
             * 
             * @param arr The array to copy from
             * @return Reference to the current tensor after assignment
             */
            tensor& operator=(const A* arr) {

                for (int i = 0; i < length(); i++) data[i] = arr[i]; // Copy the elements from the array
                return *this; // Return the current tensor
            }

            /**
             * @brief Create a tensor filled with zeros.
             * 
             * @param args Variable number of size arguments (must match tensor dimensions N)
             * @return A new tensor filled with zeros
             */
            template<typename... Args>
            static const tensor zero(Args... args) {

                tensor result(args...); // Create a new tensor to hold the result
                for (int i = 0; i < result.length(); i++) result.data[i] = 0; // Initialize all elements to zero
                return result; // Return the resulting tensor
            }

            /**
             * @brief Create an identity tensor.
             * 
             * An identity tensor has ones on the diagonal and zeros elsewhere.
             * 
             * @param args Variable number of size arguments (must match tensor dimensions N)
             * @return A new identity tensor
             * @throw "Number of size arguments must match tensor dimensions" if the number of size arguments does not match tensor dimensions
             * @throw "Identity tensor must be square" if the tensor is not square
             */
            template<typename... Args>
            static const tensor identity(Args... args) {

                // Check if the tensor is square
                int dims[] = {args...};
                if (sizeof...(args) != N) throw "Number of size arguments must match tensor dimensions";
                for (int i = 0; i < N - 1; i++)
                    if (dims[i] != dims[i + 1]) throw "Identity tensor must be square";

                tensor result(args...); // Create a new tensor to hold the result
                // Inizializza tutti a zero
                for (int i = 0; i < result.length(); i++) result.data[i] = 0;

                // Imposta a 1 solo le posizioni dove tutti gli indici sono uguali
                int size = dims[0];
                // Genera tutte le tuple (i, i, ..., i)
                for (int i = 0; i < size; ++i) {
                    
                    // Calcola l'indice lineare corrispondente a (i, i, ..., i)
                    int idx = 0;
                    for (int d = 0; d < N; ++d) idx = idx * size + i;
                    result.data[idx] = 1;
                }

                return result; // Return the resulting tensor
            }
    };

    /**
     * @brief Alias for a 2D tensor, representing a matrix.
     * 
     * This alias simplifies the usage of a 2D tensor as a matrix.
     * 
     * @tparam A Type of the matrix elements
     */
    template<typename A>
    using matrix = tensor<A, 2>;

    /**
     * @brief Alias for a 1D tensor, representing a tuple.
     * 
     * This alias simplifies the usage of a 1D tensor as a tuple.
     * 
     * @tparam A Type of the tuple elements
     */
    template<typename A>
    using tuple = tensor<A, 1>;

    /**
     * @brief Sort the elements of a tuple.
     * 
     * @tparam A Type of the tuple elements
     * @param t The tuple to sort
     * @param ascending Whether to sort in ascending order (default: true)
     * @return The sorted tuple
     */
    template<typename A>
    tuple<A> sort(tuple<A> t, bool ascending = true) {

        tuple<A> result(t);
        for (auto i = 0; i < t.size(0); i++) for (auto j = i + 1; j < t.size(0); j++)
            if ((ascending && result(i) > result(j)) || (!ascending && result(i) < result(j))) {
                
                auto temp = result(i);
                result(i) = result(j);
                result(j) = temp;
            }
        return result;
    }

    /**
     * @brief Reverse the order of elements in a tuple.
     * 
     * @tparam A Type of the tuple elements
     * @param t The tuple to reverse
     * @return The reversed tuple
     */
    template<typename A>
    tuple<A> reverse(const tuple<A>& t) {

        tuple<A> result(t.size(0));
        for (int i = 0; i < t.size(0); i++) result(i) = t(t.size(0) - 1 - i);
        return result;
    }

    /**
     * @brief Calculate the dot product of two tuples.
     * 
     * @tparam A Type of the tuple elements
     * @param a First tuple
     * @param b Second tuple
     * @return Dot product of the two tuples
     */
    template<typename A>
    A dot(const tuple<A>& a, const tuple<A>& b) {

        if (a.size(0) != b.size(0)) throw "Tuples must have the same size for dot product";
        A result = A(0);
        for (int i = 0; i < a.size(0); i++) result += a(i) * b(i);
        return result;
    }

    /**
     * @brief Calculate the dot product (element-wise multiplication) of two matrices.
     * 
     * @tparam A Type of the matrix elements
     * @param a First matrix
     * @param b Second matrix
     * @return Dot product of the two matrices
     */
    template<typename A>
    matrix<A> dot(matrix<A> a, matrix<A> b) {

        if (a.size(0) != b.size(0) || a.size(1) != b.size(1)) throw "Matrices must have the same dimensions for dot product";
        matrix<A> result(a.size(0), a.size(1));
        for (int i = 0; i < a.size(0); i++) 
            for (int j = 0; j < a.size(1); j++) result(i, j) = a(i, j) * b(i, j);
        return result;
    }

    /**
     * @brief Calculate the transpose of a matrix.
     * 
     * @tparam A Type of the matrix elements
     * @param mat Matrix to calculate the transpose of
     * @return Transpose of the matrix
     */
    template<typename A>
    matrix<A> T(matrix<A> mat) {

        matrix<A> result(mat.size(1), mat.size(0));
        for (int i = 0; i < mat.size(0); i++) 
            for (int j = 0; j < mat.size(1); j++) result(j, i) = mat(i, j);
        return result;
    }

    /**
     * @brief Calculate the trace of a matrix.
     * 
     * @tparam A Type of the matrix elements
     * @param mat Matrix to calculate the trace of
     * @return Trace of the matrix
     */
    template<typename A>
    A tr(matrix<A> mat) {

        if (mat.size(0) != mat.size(1)) throw "Matrix is not square";
        auto result = A(0);
        for (int i = 0; i < mat.size(0); i++) result += mat(i, i);
        return result;
    }

    /**
     * @brief Calculate the submatrix of a matrix, excluding one row and column.
     * 
     * @tparam A Type of the matrix elements
     * @param mat Matrix to calculate the submatrix of
     * @param excludingRow Row to exclude
     * @param excludingCol Column to exclude
     * @throw "Invalid row or column" if the row or column index is out of bounds
     * @return Submatrix of the matrix
     */
    template<typename A>
    matrix<A> submatrix(matrix<A> mat, tuple<int> del_rows, tuple<int> del_cols) {

        matrix<A> result(mat.size(0) - del_rows.size(0), mat.size(1) - del_cols.size(0));
        auto r = -1, q = 0;
        for (auto i = 0; i < mat.size(0); i++) {

            for (auto j = 0; j < del_rows.size(0); j++) if (i == del_rows(j)) q = 1;
            if (q) { q = 0; continue; }
            r++;
            auto c = -1;
            for (auto j = 0; j < mat.size(1); j++) {

                for (auto k = 0; k < del_cols.size(0); k++) if (j == del_cols(k)) q = 1;
                if (q) { q = 0; continue; }
                c++;
                result(r, c) = mat(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Calculate the adjugate of a matrix.
     * 
     * @tparam A Type of the matrix elements
     * @param mat Matrix to calculate the adjugate of
     * @return Adjugate of the matrix
     */
    template<typename A>
    matrix<A> adj(matrix<A> mat) {

        if (mat.size(0) != mat.size(1)) throw "Matrix is not square";
        matrix<A> result(mat.size(0), mat.size(1));
        for (auto i = 0; i < mat.size(0); i++)
            for (auto j = 0; j < mat.size(1); j++) {

                tuple<int> del_row(1); del_row(0) = i;
                tuple<int> del_col(1); del_col(0) = j;
                auto subm = submatrix(mat, del_row, del_col);
                result(i, j) = det(subm) * ((i + j) % 2 == 0 ? (A)1 : (A)(-1));
            }
        return result;
    }

    /**
     * @brief Calculate the determinant of a matrix.
     * 
     * @tparam A Type of the matrix elements
     * @param mat Matrix to calculate the determinant of
     * @return Determinant of the matrix
     */
    template<typename A>
    A det(matrix<A> mat) {

        if (mat.size(0) != mat.size(1)) throw "Matrix is not square";
        else if (mat.size(0) == 1) return mat(0, 0);
        else {

            auto res = A(0);
            for (auto p = 0; p < mat.size(0); p++) {

                tuple<int> del_row(1); del_row(0) = 0;
                tuple<int> del_col(1); del_col(0) = p;
                res += mat(0, p) * det(submatrix(mat, del_row, del_col)) * ((p % 2 == 0) ? A(1) : A(-1));
            }
            return res;
        }
    }
}

#endif