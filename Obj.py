import copy
import pandas as pd
from collections import Counter
from podm import BoundingBox, get_pascal_voc_metrics, MetricPerClass


class ObjDetectionMetricHelper(object):
    __requiredKeys = ['model_type', 'model_name', 'model_version']
    mertics = None
    KPI = 'mAP'
    KPI_value = 0
    cm = {}
    bboxes = {"GT": [],
              "IFR": []}

    def __init__(self, faMetric):
        self.cm = {key: faMetric[key] for key in self.__requiredKeys}
        dfRows = []
        if set(list(self.cm.keys())) != set(self.__requiredKeys):
            raise ValueError("Keys:{} are not fully contained in FA data.".format(self.__requiredKeys))
        faV = faMetric.get("values")
        if faV is None:
            raise ValueError("'values' is not found in FA metric.")
        self._calImgMetric(faV)
        self._mapping_dataframe()
        #

        #for value in faV:
          #  pn, imgBboxes = self._calImgMetric(value)
           # self.bboxes["GT"] += imgBboxes["GT"]
            #self.bboxes["IFR"] += imgBboxes["IFR"]
            #for k, v in pn.items():
             #   df_row = copy.deepcopy(v)
        #         df_row.update({"category": k})
        #         dfRows.append(df_row)
        #     value.update({"additional": pn})
        #
        # self.fa = faMetric
        # self.imgDF = pd.DataFrame(dfRows, columns=["category", "tp", "fp", "fn"])

    def _mapping_dataframe(self):
        df_rows = []

        for category in self.metrics:
            tp_array = self.metrics[category].array_tp
            fp_array = self.metrics[category].array_fp
            for array_name in [tp_array, fp_array]:
                for image_name, contents in array_name.items():
                    for c in contents:
                        rows = {}
                        bbox = [c.xtl, c.ytl, c.xbr, c.ybr]
                        rows.update({"image_name": image_name, "category": category, "IOU": c.iou, "index": c.index, "bbox": bbox, "category_probs": c.cate_probs})
                        df_rows.append(rows)

        self.data_frame = pd.DataFrame(df_rows, columns=["image_name", "category", "IOU", "index", "bbox", "category_probs"])
        # Sorting by IOU
        self.data_frame.sort_values(by=['IOU'], inplace=True, ascending=False)

        print(self.data_frame)
        # Map fp index
        print(self.GT_df)


    def _calImgMetric(self, values):
        # value is dict of {"GT":[], "IFR":[], "file_name":"", "file_path"}
        boundingbox = {}
        boundingbox["GT"] = []
        boundingbox["IFR"] = []
        gt_df_rows = []

        for value in values:
            file_name = value.get("file_name", "")

            for pn in ["GT", "IFR"]:
                bbox_list = value[pn]
                index_setting = []

                for box in bbox_list:
                    bbox = box.get("bbox")
                    label = box.get("category_name")
                    cate_probs = box.get("category_probs")

                    if pn == "GT":
                        if not label in index_setting:
                            index = 1
                            index_setting.append(label)
                        else:
                            index += 1

                        rows = {}
                        rows.update({"image_name": file_name, "category": label, "index": index})
                        gt_df_rows.append(rows)
                    else:
                        index = -1

                    box.update({"index": index})
                    try:
                        score = max(box.get("category_probs"))
                    except:
                        score = None
                    # BoundingBox(file, label, x1, y1, x2, y2, score)
                    brx = bbox[0] + bbox[2]  # x of button right point
                    bry = bbox[1] + bbox[3]  # y of button right point
                    boundingbox[pn].append(BoundingBox(file_name, label, bbox[0], bbox[1], brx, bry, score, index, cate_probs=cate_probs))

            self.GT_df = pd.DataFrame(gt_df_rows, columns=["image_name", "category", "index"])
            self.metrics = get_pascal_voc_metrics(boundingbox["GT"], boundingbox["IFR"], tpfp_only=False)
            # pn = {}
            # for category in self.metrics:
            #     print(category)
                # for image_name, boundbox in self.metrics[category].array_tp.items():
                #     if image_name == value['file_name']:
                #         tp = len(boundbox)
                #
                # for image_name, boundbox in self.metrics[category].array_fp.items():
                #     if image_name == value['file_name']:
                #         fp = len(boundbox)
                #
                # for image_name, boundbox in self.metrics[category].array_fn.items():
                #     if image_name == value['file_name']:
                #         fn = len(boundbox) - tp
                #
                # pn.update({category: {"tp": tp, "fp": fp, "fn": fn}})

            # value.update({"additional": pn})
            # value.update({"results": {}})
            # print(value)

    def getCM(self):
        # print(self.imgDF)
        # [confusion_matrix, ap_by_class, IOU, precision_recall]
        # confusion_matrix
        categoryAggDf = self.imgDF.groupby(by=['category']).sum().groupby(level=[0]).cumsum()
        # print(categoryAggDf)
        table_name = list(categoryAggDf.index)
        table_value = []
        for index, row in categoryAggDf.iterrows():
            table_value.append({"name": index,
                                "value": [[row["fp"], 0], [row["tp"], row["fn"]]]})
        self.cm.update({"confusion_matrix": {"table_name": table_name,
                                             "table_type": "confusion_matrix",
                                             "x-axis": ["P", "N"],
                                             "y-axis": ["N", "P"],
                                             "tables_value": table_value}})

        # ap_by_class
        values = []
        categories = []
        tables_value_ap = []
        for category in self.mertics:
            values.append(self.metrics[category].ap)
            categories.append(category)
        tables_value_ap.append({"name": "AP_Overall", "value": values})
        self.cm.update({"ap_by_class": {
            "table_name": ["AP_Overall"],
            "table_type": "bar_plot",
            "tables_value": tables_value_ap,
            "x-axis": categories}})

        # IOU
        tables_value_iou = []
        for category in self.mertics:
            content = {}
            return_values = []
            for _, values in self.metrics[category].tp_pair.items():
                for value in values:
                    return_values.append(value[2])
            content.update({"name": category, "value": return_values})
            tables_value_iou.append(content)
        self.cm.update({"IOU": {
            "table_name": ["IOU_Overall"],
            "table_type": "histogram_plot",
            "tables_value": tables_value_iou}})

        # precision_recall
        precision_content = {}
        recall_content = {}
        f1_score_content = {}
        for category in self.metrics:
            tp = self.metrics[category].tp
            fp = self.metrics[category].fp
            fn = self.metrics[category].num_groundtruth - tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = (2 * precision * recall) / (precision + recall)
            precision_content.update({category: precision})
            recall_content.update({category: recall})
            f1_score_content.update({category: f1_score})

        self.cm.update({"precision_recall": {"results": {
            "Precision": precision_content,
            "Recall": recall_content,
            "F1-Score": f1_score_content}}})

        return self.cm

    def getFA(self):
        return self.fa

    def getKPI(self):
        metrics = get_pascal_voc_metrics(self.bboxes["GT"], self.bboxes["IFR"])
        return {self.KPI: MetricPerClass.mAP(metrics)}

