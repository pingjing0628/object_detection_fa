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
        if set(list(self.cm.keys())) != set(self.__requiredKeys):
            raise ValueError("Keys:{} are not fully contained in FA data.".format(self.__requiredKeys))
        faV = faMetric.get("values")
        if faV is None:
            raise ValueError("'values' is not found in FA metric.")
        self._calImgMetric(faV)

    def _process_FA(self, values, gt_category_idx):
        for v in values:
            # delete the id in box
            for box in v['GT']:
                del box['id']
            # update index in IFR and delete id
            for box in v['IFR']:
                # overwrite all index from DF
                box['index'] = self.voc_df.loc[self.voc_df['id'] == box['id']]['index'].values[0]
                del box['id']

            # deal additional by GT
            f_df = self.voc_df.loc[self.voc_df['filename'] == v['file_name']]
            category_list = list(f_df['category'].unique())
            img_additional = {}
            img_results = {}
            for cat in category_list:
                # start to deal additional by category
                f_c_df = f_df.loc[f_df['category'] == cat]
                gt = gt_category_idx[v['file_name']].get(cat, 0)
                tp = f_c_df.loc[f_c_df['tag']=='tp'].shape[0]
                fp = f_c_df.loc[f_c_df['tag']=='fp'].shape[0]
                img_additional[cat] = {'tp': tp,
                                       'fp': fp,
                                       'fn': gt - tp}

                # start to deal results by category
                # get tp,fp which use index of GT
                gt_index_results_df = f_c_df.loc[f_c_df['index']<=gt]
                cat_box_results = []
                for idx, row in gt_index_results_df.iterrows():
                    cat_box_results.append({'index': row['index'], 'IOU':row['iou']})
                img_results[cat] = cat_box_results
            v['additional'] = img_additional

            # deal results
            v['results'] = img_results
        return values

    def _VOC2DF(self, gt_category_idx):
        # convert TP, FP to DF
        data = []

        # Set data
        def put_data(cate, data, array_xp, tag):
            for file_name, bboxs in array_xp.items():
                for bbox in bboxs:
                    data.append({'tag': tag,
                                 'category': cate,
                                 'index': bbox.index,
                                 'filename': file_name,
                                 'id': bbox.id,
                                 'iou': bbox.iou})

        # Merge tp fp array in one dataframe for mapping
        for category in self.metrics:
            put_data(category, data, self.metrics[category].array_tp, 'tp')
            put_data(category, data, self.metrics[category].array_fp, 'fp')

        df = pd.DataFrame(data)
        print(df)
        # Deal index of fp for index setting
        # List all file name
        filename_list = list(df['filename'].unique())
        for filename in filename_list:
            # df.loc is for conditions
            # 1. filter with file name
            f_df = df.loc[df['filename'] == filename]
            # 2. list all category
            file_cat_list = list(f_df['category'].unique())

            for c in file_cat_list:
                # 3. filter with category
                f_c_df = f_df.loc[f_df['category'] == c]
                # 4. sorting by iou
                f_c_df = f_c_df.sort_values(by=['iou'], ascending=False)
                # 5. count fp , shape[0] is count the height
                fp_count = f_c_df[f_c_df['tag'] == 'fp'].shape[0]

                # initialize mapping idx list for GT, extra_idx
                # 6. get max gt idx
                max_gt_category_idx = gt_category_idx[filename].get(c, 0)
                # 7. make a list for gt idx
                gt_idxs = list(range(1, max_gt_category_idx + 1))
                # 8. make a list for fp index modified, range from max gt +1 to max gt +1 + fp count
                extra_idxs = list(range(max_gt_category_idx + 1, max_gt_category_idx + 1 + fp_count))

                # check index and assign it to right value
                for idx, row in f_c_df.iterrows():
                    # 9. find the index in gt, then remove from gi_idx list
                    if row['index'] in gt_idxs:
                        gt_idxs.remove(row['index'])
                    # df.loc[行, 列]
                    # 10. replace index to the key pop from extra_idxs list
                    else:
                        df.loc[idx, 'index'] = extra_idxs.pop(0)

        return df

    def _calImgMetric(self, values):
        # value is dict of {"GT":[], "IFR":[], "file_name":"", "file_path"}
        boundingbox = {}
        boundingbox["GT"] = []
        boundingbox["IFR"] = []
        box_id = 1
        gt_category_idx = {}
        '''
        {'filename':{'cat1': 2, 'cat2': 1}
        '''
        for value in values:
            file_name = value.get("file_name", "")

            # initial category idx
            gt_category_idx[file_name] = {}

            for pn in ["GT", "IFR"]:
                bbox_list = value[pn]
                for box in bbox_list:
                    bbox = box.get("bbox")
                    label = box.get("category_name")
                    # For default setting all values have index and box id
                    box.update({'index': -1, 'id': box_id})

                    if pn == "GT":
                        label_idx = gt_category_idx[file_name]

                        # count category
                        if label in label_idx.keys():
                            label_idx[label] += 1
                        else:
                            label_idx[label] = 1

                        # update same category to add 1
                        box['index'] = label_idx[label]

                    try:
                        score = max(box.get("category_probs"))
                    except:
                        score = None

                    # BoundingBox(file, label, x1, y1, x2, y2, score)
                    brx = bbox[0] + bbox[2]  # x of button right point
                    bry = bbox[1] + bbox[3]  # y of button right point
                    boundingbox[pn].append(BoundingBox(file_name, label, bbox[0], bbox[1], brx, bry, score,
                                                       index=box['index'], id=box['id']))
                    box_id += 1

        self.metrics = get_pascal_voc_metrics(boundingbox["GT"], boundingbox["IFR"])

        # Let bring gt dict to use in mapping tp fp dataframe
        self.voc_df = self._VOC2DF(gt_category_idx)

        # process GT df
        gt_df = pd.DataFrame(gt_category_idx.values())
        # fill 0 which the field is NaN
        gt_df = gt_df.fillna(0)

        self.gt_count = {}
        # want to sum the value in each column
        for col in gt_df.columns:
            self.gt_count.update({col: gt_df[col].sum()})

        # for debug
        print(self.voc_df.head(10))

        # make FA
        self.fa = self._process_FA(values, gt_category_idx)

        return {}, []

    def getCM(self):
        # [confusion_matrix, ap_by_class, IOU, precision_recall]
        cm_table_value = []
        cm_table_name = []
        confusion = []
        df_rows = []
        ap_values = []
        ap_categories = []
        tables_value_ap = []
        tables_value_iou = []
        precision_content = {}
        recall_content = {}
        f1_score_content = {}

        confusion_df = self.voc_df.groupby(["category", "tag"], as_index=False)['id'].count()
        confusion_df.rename(columns={'id': 'count'}, inplace=True)
        #print(confusion_df)
        #print(self.gt_count)

        confusion_df = confusion_df.reset_index().groupby(['category', 'tag'])['count'].aggregate('first').unstack()
        confusion_df = confusion_df.fillna(0)
        for key in self.gt_count:
            cm_table_name.append(key)

        for index, row in confusion_df.iterrows():
            fn = int(self.gt_count[index] - row["tp"])
            cm_table_value.append({"name": index,
                                   "value": [[row["fp"], 0], [row["tp"], fn]]})

        self.cm.update({"confusion_matrix": {"table_name": cm_table_name,
                                             "table_type": "confusion_matrix",
                                             "x-axis": ["P", "N"],
                                             "y-axis": ["N", "P"],
                                             "tables_value": cm_table_value}})

        for category in self.metrics:
            # ap_by_class
            ap_values.append(self.metrics[category].ap)
            ap_categories.append(category)

            # IOU
            iou_content = {}
            iou_return_values = []
            for image_name, values in self.metrics[category].array_tp.items():
                for value in values:
                    iou_return_values.append(value.iou)
            iou_content.update({"name": category, "value": iou_return_values})
            tables_value_iou.append(iou_content)

            # precision_recall
            tp = self.metrics[category].tp
            fp = self.metrics[category].fp
            fn = self.metrics[category].num_groundtruth - tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_base = precision + recall
            if f1_base == 0:
                f1_score = 0
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
            precision_content.update({category: precision})
            recall_content.update({category: recall})
            f1_score_content.update({category: f1_score})

        tables_value_ap.append({"name": "AP_Overall", "value": ap_values})
        self.cm.update({"ap_by_class": {"table_name": ["AP_Overall"],
                                        "table_type": "bar_plot",
                                        "tables_value": tables_value_ap,
                                        "x-axis": ap_categories}})
        self.cm.update({"IOU": {"table_name": ["IOU_Overall"],
                                "table_type": "histogram_plot",
                                "tables_value": tables_value_iou}})
        self.cm.update({"precision_recall": {
                        "results": {
                            "Precision": precision_content,
                            "Recall": recall_content,
                            "F1-Score": f1_score_content
                        }}})
        return self.cm

    def getFA(self):
        return self.fa

    def getKPI(self):
        return {self.KPI: MetricPerClass.mAP(self.metrics)}