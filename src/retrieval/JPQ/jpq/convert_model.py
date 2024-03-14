import argparse
import os

import faiss
import torch
from transformers import RobertaConfig

from .model import JPQTower


def format_encoder(
    robertadot_path: str, init_index_path: str, output_path: str
) -> None:
    """
    Converts RobertaDot model to JPQTower model
    source: https://github.com/jingtaozhan/JPQ/issues/4
    """
    opq_index = faiss.read_index(init_index_path)

    vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))
    assert isinstance(vt, faiss.LinearTransform)
    opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

    ivf_index = faiss.downcast_index(opq_index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)

    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
    centroid_embeds = centroid_embeds.reshape(
        ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub
    )
    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
    coarse_embeds[:] = 0

    config = RobertaConfig.from_pretrained(robertadot_path)
    config.name_or_path = output_path
    config.MCQ_M, config.MCQ_K = ivf_index.pq.M, ivf_index.pq.ksub

    model = JPQTower.from_pretrained(robertadot_path, config=config)

    with torch.no_grad():
        model.centroids.copy_(torch.from_numpy(centroid_embeds))
        model.rotation.copy_(torch.from_numpy(opq_transform))

    model.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=int, required=True)
    parser.add_argument("--subvector_num", type=int, required=True)
    parser.add_argument("--enwiki_name", default=None, type=str, required=True)
    args = parser.parse_args()

    data_type = "doc" if args.data_type == 0 else "passage"
    m = args.subvector_num
    index_name = f"OPQ{m},IVF1,PQ{m}x8.index"
    jpq_path = (
        f"data/{data_type}/eval/{args.enwiki_name}/m{m}/doc_encoder",
        f"data/{data_type}/eval/{args.enwiki_name}/m{m}/query_encoder",
    )

    # Create directories
    os.makedirs(jpq_path[0], exist_ok=True)
    os.makedirs(jpq_path[1], exist_ok=True)

    # Convert Document encoder
    format_encoder(
        f"data/{data_type}/star",
        f"data/{data_type}/init/m{m}/{index_name}",
        jpq_path[1],
    )

    # Convert Query encoder
    q_model = f"data/{data_type}/train/m{m}/models/epoch-6"
    format_encoder(q_model, f"{q_model}/{index_name}", jpq_path[0])


if __name__ == "__main__":
    main()
