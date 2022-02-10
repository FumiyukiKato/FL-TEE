import grpc

import secure_aggregation_pb2
import secure_aggregation_pb2_grpc


ADDRESS = '127.0.0.1:50051'
COUNTER_LEN = 16


def call_grpc(
        encrypted_parameters,
        num_of_parameters,
        num_of_sparse_parameters,
        client_ids,
        sigma,
        clipping,
        alpha,
        aggregation_alg):
    with grpc.insecure_channel(ADDRESS) as channel:
        stub = secure_aggregation_pb2_grpc.AggregatorStub(channel)
        response = stub.UpdateParameters(
            secure_aggregation_pb2.ParametersRequest(
                encrypted_parameters=bytes(encrypted_parameters),
                num_of_parameters=num_of_parameters,
                num_of_sparse_parameters=num_of_sparse_parameters,
                client_ids=client_ids,
                sigma=sigma,
                clipping=clipping,
                alpha=alpha,
                aggregation_alg=aggregation_alg
            )
        )
    return response.updated_parameters, float(response.execution_time)
