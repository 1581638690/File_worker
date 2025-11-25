def extract_ocr_info_v2(ocr_output):
    """适配你真实 JSON 结构的 OCR 解析器"""
    result = {
        "full_text": "",
        "blocks": [],
        "layout_boxes": [],
        "seal_candidates": []
    }

    if not ocr_output:
        return result

    #data = ocr_output.get("res",{})
    data = ocr_output
    # --- 文本块 ---
    for blk in data.get("parsing_res_list", []):
        content = blk.get("block_content") or ""
        label   = blk.get("block_label") or ""
        bbox    = blk.get("block_bbox", [])
        blk_id  = blk.get("block_id", None)
        order   = blk.get("block_order", None)

        if content:
            result["full_text"] += content + "\n"

        result["blocks"].append({
            "content": content,
            "label": label,
            "bbox": bbox,
            "id": blk_id,
            "order": order
        })

    # --- 布局框 ---
    layout = data.get("layout_det_res", {})
    for box in layout.get("boxes", []):
        label = box.get("label", "")
        score = box.get("score", 0)
        coord = box.get("coordinate", [])

        result["layout_boxes"].append({
            "label": label,
            "score": score,
            "bbox": coord
        })

        if label == "figure":
            result["seal_candidates"].append(coord)

    return result


def detect_stamp_from_image(image_path,SEAL_PIPE):
    """
    对单张图片执行印章检测，返回印章文字及分数
    """
    try:
        output = SEAL_PIPE.predict(
            image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False

        )
        stamp_texts = []

        for res in output:
            data = res._to_json()
            #data_dic = data.get("res",{})
            data_dic = data
            seal_list = data_dic.get("seal_res_list",[])
            for seal in seal_list:
                texts = seal.get("rec_texts",[])
                scores = seal.get("rec_scores", [])
                for txt, score in zip(texts, scores):
                    stamp_texts.append({
                        "text": txt,
                        "score": score,
                        #"poly": seal.get("dt_polys", [])
                    })
        return stamp_texts

    except:
        return []
    

def detect_stamp_from_image_new(output):
    """
    对单张图片执行印章检测，返回印章文字及分数
    """
    try:
        stamp_texts = []



            #data_dic = data.get("res",{})
        data_dic = output
        seal_list = data_dic.get("seal_res_list",[])
        for seal in seal_list:
            texts = seal.get("rec_texts",[])
            scores = seal.get("rec_scores", [])
            for txt, score in zip(texts, scores):
                stamp_texts.append({
                    "text": txt,
                    "score": score,
                    #"poly": seal.get("dt_polys", [])
                })
        return stamp_texts

    except:
        return []