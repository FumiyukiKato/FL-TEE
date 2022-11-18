import grpc

import secure_aggregation_pb2
import secure_aggregation_pb2_grpc


ADDRESS = '127.0.0.1:50051'
COUNTER_LEN = 16


def call_grpc_aggregate(
    fl_id,
    round,
    encrypted_parameters,
    num_of_parameters,
    num_of_sparse_parameters,
    client_ids,
    aggregation_alg,
    optimal_num_of_clients,
):
    with grpc.insecure_channel(ADDRESS) as channel:
        stub = secure_aggregation_pb2_grpc.AggregatorStub(channel)
        response = stub.Aggregate(
            secure_aggregation_pb2.AggregateRequestParameters(
                fl_id=fl_id,
                round=round,
                encrypted_parameters=bytes(encrypted_parameters),
                num_of_parameters=num_of_parameters,
                num_of_sparse_parameters=num_of_sparse_parameters,
                aggregation_alg=aggregation_alg,
                optimal_num_of_clients=optimal_num_of_clients,
                client_ids=client_ids,
            )
        )
    return response.updated_parameters, float(response.execution_time), response.client_ids, response.round

def call_grpc_start(
    fl_id: int,
    client_ids: list[int],
    sigma: float,
    clipping: float,
    alpha: float,
    sampling_ratio: float,
    aggregation_alg: int,
    num_of_parameters: int,
    num_of_sparse_parameters: int,
):
    with grpc.insecure_channel(ADDRESS) as channel:
        stub = secure_aggregation_pb2_grpc.AggregatorStub(channel)
        response = stub.Start(
            secure_aggregation_pb2.StartRequestParameters(
                fl_id=fl_id,
                client_ids=client_ids,
                sigma=sigma,
                clipping=clipping,
                alpha=alpha,
                sampling_ratio=sampling_ratio,
                aggregation_alg=aggregation_alg,
                num_of_parameters=num_of_parameters,
                num_of_sparse_parameters=num_of_sparse_parameters,
            )
        )
    return response.fl_id, response.round, response.client_ids
