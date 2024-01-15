import pennylane as qml
from pennylane import numpy as np

class QCNN:

    def __init__(self, n_qubits, conv_ansatz, pooling_ansatz, stride):
        self.conv_ansatz = conv_ansatz
        self.pooling_ansatz = pooling_ansatz
        self.n_qubits = n_qubits
        self.stride = stride
        # total required layer for QCNN
        self.total_layer = 1
        # information about used qubits in each layer
        self.QCNN_tree = []
        # ancilla qubit location
        self.ancilla_info = []
        # do calculation
        temp_q = self.n_qubits
        temp_qs = list(range(self.n_qubits))
        temp_ancilla = self.n_qubits
        while (temp_q != 1):
            # does not require ancilla
            if temp_q % 2 == 0:
                self.QCNN_tree.append(temp_qs.copy())
                self.ancilla_info.append(-1)
                temp_qs = [int(qs)
                           for (idx, qs) in enumerate(temp_qs) if idx % 2 == 0]
                temp_q = int(temp_q/2)
            # require ancilla
            else:
                self.ancilla_info.append(temp_ancilla)
                temp_qs.append(temp_ancilla)
                temp_ancilla += 1
                self.QCNN_tree.append(temp_qs.copy())
                temp_qs = [int(qs)
                           for (idx, qs) in enumerate(temp_qs) if idx % 2 == 0]
                temp_q = int((temp_q+1)/2)
            self.total_layer += 1

        self.QCNN_tree.append([0])
        self.ancilla_info.append(-1)
        # total qubit required
        self.total_qubit = temp_ancilla

    def Calculate_Param_Num(self):
        ansatz_param = self.conv_ansatz.num_params
        pooling_param = self.pooling_ansatz.num_params
        total_param_num = 0
        for layer in range(self.total_layer):
            qubit_info = np.array(self.QCNN_tree[layer])
            L = len(qubit_info)
            qubit_info_index = np.array(range(L))
            if self.stride == 1:
                qubit_info_splited = [[idx for idx in qubit_info_index if idx % 2 == 0], [
                    idx for idx in qubit_info_index if idx % 2 == 1]]
            else:
                qubit_info_splited = qubit_info_index
            # apply unitary
            for index in qubit_info_splited:
                for idx in index:
                    if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                        total_param_num += ansatz_param
            # apply pooling
            for idx in qubit_info_index:
                if idx % 2 == 0:
                    if qubit_info[idx] != qubit_info[(idx+1) % L]:
                        total_param_num += pooling_param
        return total_param_num

    def construct_circuit(self, thetas, data):
        # insert initial state as data in n_qubits, not total_qubits
        qml.AmplitudeEmbedding(data, wires=range(self.n_qubits), pad_with=0, normalize=True)
        theta_idx = 0
        for layer in range(self.total_layer):
            qubit_info = list(self.QCNN_tree[layer])
            L = len(qubit_info)
            qubit_info_index = list(range(L))
            if self.stride == 1:
                qubit_info_splited = [[idx for idx in qubit_info_index if idx % 2 == 0], [
                    idx for idx in qubit_info_index if idx % 2 == 1]]
            else:
                qubit_info_splited = qubit_info_index
            # apply convolution
            for index in qubit_info_splited:
                for idx in index:
                    if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                        self.conv_ansatz.apply(
                            thetas[theta_idx:theta_idx+self.conv_ansatz.num_params], wires=[qubit_info[idx], qubit_info[(idx+self.stride) % L]])
                        theta_idx += self.conv_ansatz.num_params
            # apply pooling
            for idx in qubit_info_index:
                if idx % 2 == 0:
                    if qubit_info[idx] != qubit_info[(idx+1) % L]:
                        self.pooling_ansatz.apply(thetas[theta_idx:theta_idx+self.pooling_ansatz.num_params],
                                wires=[qubit_info[(idx+1) % L], qubit_info[idx]])
                        theta_idx += self.pooling_ansatz.num_params