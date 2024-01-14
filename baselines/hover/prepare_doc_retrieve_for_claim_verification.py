import argparse
from itertools import chain
import os
import json
import logging
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(sys.path[0] + '/../..')
from scripts.monitor_utils import monitor

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_split",
        default=None,
        type=str,
        required=True,
        help="[train | dev | test]",
    )
    parser.add_argument(
        "--doc_retrieve_range",
        default=5,
        type=int,
        help="Top k retrieved documents to be used for claim verification."
    )
    parser.add_argument(
        "--data_dir",
        default='data',
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default='hover',
        type=str
    )
    parser.add_argument(
        "--doc_retrieval_output_dir",
        default="exp1.0",
        type=str,
    )
    parser.add_argument(
        "--doc_retrieval_model_global_step",
        default=900,
        type=int
    )
    parser.add_argument(
        "--rerank_mode",
        default="none",
        type=str,
        help="[none | within | between]"
    )
    parser.add_argument(
        "--top_k",
        default=5,
        type=int,
        help="Top-k sentences to retrieve for reranking"
    )
    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.dataset_name)
    doc_data = json.load(open(os.path.join(args.data_dir, 'doc_retrieval', 'hover_' + args.data_split+'_doc_retrieval.json')))
    args.doc_retrieval_output_dir = os.path.join('out', args.dataset_name, args.doc_retrieval_output_dir, 'doc_retrieval', \
        'checkpoint-'+str(args.doc_retrieval_model_global_step))
    doc_retrieval_predictions = json.load(open(os.path.join(args.doc_retrieval_output_dir, args.data_split+'_predictions_.json')))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )

    uid_to_doc = {}
    for e in doc_data:
        uid, claim, label, context = e['uid'], e['claim'], e['label'], e['context']
        assert uid not in uid_to_doc
        uid_to_doc[uid] = [claim, label, context]

    # Get top-k predicted titles for every claim
    data_for_claim_verif = []
    for uid in tqdm(doc_retrieval_predictions.keys()):
        context = []
        if uid in uid_to_doc:
            pred_titles = []
            sorted_titles = doc_retrieval_predictions[uid]['sorted_titles']
            topk = min(len(sorted_titles), args.doc_retrieve_range)
            for idx in range(topk):
                pred_titles.append(sorted_titles[idx])

            claim, label, doc_context = uid_to_doc[uid]
            # Check if the doc data paragraph title is in the top-k titles
            if args.rerank_mode == "none":
                context = " ".join([" ".join(paragraph[1:]) for paragraph in doc_context if paragraph and paragraph[0] in pred_titles])
            elif args.rerank_mode == "within":
                with torch.no_grad():
                    claim_emb = encoder.encode(claim)
                    for paragraph in doc_context:
                        if paragraph and paragraph[0] in pred_titles:
                            doc_sents = paragraph[1:]
                            sent_embeds = encoder.encode(paragraph[1:])
                            similarity_scores = cosine_similarity([claim_emb], sent_embeds)[0]
                            # sort on similarity scores and afterwards sort top-k sentences back in original order.
                            sorted_sents = [sent for _, sent in sorted(zip(similarity_scores, enumerate(doc_sents)), key=lambda x: x[0], reverse=True)]
                            top_k_sents = " ".join([sent for _, sent in sorted(sorted_sents[:args.top_k], key=lambda x: x[0])])
                            context.append(top_k_sents)
                context = " ".join(context)
            else:
                doc_sents = list(chain.from_iterable([paragraph[1:] for paragraph in doc_context 
                                                        if paragraph and paragraph[0] in pred_titles]))
                with torch.no_grad():
                    claim_emb = encoder.encode(claim)
                    sent_embeds = encoder.encode(doc_sents)
                similarity_scores = cosine_similarity([claim_emb], sent_embeds)[0]
                sorted_sents = [sent for _, sent in sorted(zip(similarity_scores, enumerate(doc_sents)), key=lambda x: x[0], reverse=True)]
                context = [sent for _, sent in sorted(sorted_sents[:args.top_k], key=lambda x: x[0])]
                context = " ".join(context)

            dp = {'id': uid, 'claim': claim, 'context': context, 'label': label}
            data_for_claim_verif.append(dp)

    logging.info("Saving prepared data ...")
    with open(os.path.join(args.data_dir, 'claim_verification', 'hover_'+args.data_split+'_claim_verification.json'), 'w', encoding="utf-8") as f:
        json.dump(data_for_claim_verif, f)

if __name__ == "__main__":
    monitor(main)