import pennylane as qml
from pennylane import numpy as np

class QCNN:

    def __init__(self, n_qubits, conv_ansatz, stride):
        self.conv_ansatz = conv_ansatz
        # no pooling layer
        self.n_qubits = n_qubits
        self.stride = stride
        # total required layer for QCNN
        self.total_layer = 1
        # information about used qubits in each layer
        self.QCNN_tree = []
        # do calculation
        temp_q = self.n_qubits
        temp_qs = list(range(self.n_qubits))
        #total qubit number
        self.total_qubit = self.n_qubits
        while (temp_q != 1):
            # does not skip pooling
            if temp_q % 2 == 0:
                self.QCNN_tree.append(temp_qs.copy())
                temp_qs = [int(qs)
                    for (idx, qs) in enumerate(temp_qs) if idx % 2 == 0]
                temp_q = int(temp_q/2)
            # skip_pooling
            else:
                self.QCNN_tree.append(temp_qs.copy())
                temp_qs = [int(qs)
                    for (idx, qs) in enumerate(temp_qs) if idx % 2 == 0]
                temp_q = int((temp_q+1)/2)
            self.total_layer += 1

        self.QCNN_tree.append([0])

    def Calculate_Param_Num(self):
        ansatz_param = self.conv_ansatz.num_params
        # no pooling param
        total_param_num = 0
        for layer in range(self.total_layer):
            qubit_info = np.array(self.QCNN_tree[layer])
            L = len(qubit_info)
            qubit_info_index = np.array(range(L))
            if self.stride == 1:
                qubit_info_splitted = [[idx for idx in qubit_info_index if idx % 2 == 0],
                    [idx for idx in qubit_info_index if idx % 2 == 1]]
            else:
                qubit_info_splitted = qubit_info_index
            # apply unitary
            for index in qubit_info_splitted:
                if L > 2:
                    for idx in index:
                        if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                            total_param_num += ansatz_param
                else:
                    idx = qubit_info_splitted[0][0]
                    if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                        total_param_num += ansatz_param
            # no pooling ansatz

        return total_param_num

    def construct_circuit(self, thetas, data):
        # insert initial state as data in total_qubits
        qml.AmplitudeEmbedding(data, wires=range(self.n_qubits), pad_with=0, normalize=True)
        theta_idx = 0
        for layer in range(self.total_layer):
            qubit_info = list(self.QCNN_tree[layer])
            L = len(qubit_info)
            qubit_info_index = list(range(L))
            if self.stride == 1:
                qubit_info_splitted = [[idx for idx in qubit_info_index if idx % 2 == 0],
                    [idx for idx in qubit_info_index if idx % 2 == 1]]
            else:
                qubit_info_splitted = qubit_info_index
            # apply convolution
            for index in qubit_info_splitted:
                if L > 2:
                    for idx in index:
                        if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                            self.conv_ansatz.apply(
                                thetas[theta_idx:theta_idx+self.conv_ansatz.num_params], wires=[qubit_info[idx], qubit_info[(idx+self.stride)%L]])
                            theta_idx += self.conv_ansatz.num_params
                else:
                    idx = qubit_info_splitted[0][0]
                    if qubit_info[idx] != qubit_info[(idx+self.stride) % L]:
                        self.conv_ansatz.apply(
                            thetas[theta_idx:theta_idx+self.conv_ansatz.num_params], wires=[qubit_info[idx], qubit_info[(idx+self.stride)%L]])

            # no separate pooling layer