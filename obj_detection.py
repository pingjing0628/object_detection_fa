from Obj1 import ObjDetectionMetricHelper

input_json = {
  "model_name": "ObjectDetection_fa",
  "model_type": "ObjectDetection",
  "model_version": "v0.0.0.18",
  "image_removed": "false",
  "values": [
    {
      "GT": [
        {
          # "index": "1",
          "category_name": "car",
          "bbox": [
                16,
                20,
                20,
                60
            ]
        },
        {
          # "index" : "2",
          "category_name": "car",
          "bbox": [
                33,
                50,
                50,
                60
            ]
        },
        {
          # "index" : "1",
          "category_name" : "person",
          "bbox": [
                16,
                20,
                20,
                60
            ]
        },
        {
          # "index" : "2",
          "category_name" : "person",
          "bbox": [
                16,
                20,
                20,
                60
            ]
        }
      ],
      "IFR": [
        {
          # "index" : "3",
          "category_name" : "car",
          "bbox": [
                50,
                70,
                70,
                80
            ],
          "category_probs": [
                0.2,
                0.3
            ]
        },
        {
          # "index" : "2",
          "category_name" : "car",
          "bbox": [
                33,
                50,
                50,
                60
            ],
          "category_probs": [
                0.2,
                0.3
            ]
        },
        {
          # "index" : "1",
          "category_name" : "car",
          "bbox": [
                16,
                20,
                20,
                60
            ],
          "category_probs": [
                0.2,
                0.3
            ]
        },

        {
          # "index" : "2",
          "category_name" : "person",
          "bbox": [
                16,
                20,
                20,
                60
            ],
          "category_probs": [
                0.2,
                0.3
            ]
        }
      ],
      "file_name": "1.png",
      "file_path": "/project/e1f4d70d6d377d017bca6f4617de72fa/pipeline/keras-densenet-classifier_v0.20.0.16/20201029061557/1.png",
      "meta": {
        "dataset_name": "val_only_aoi",
        "dataset_token": "55ede3ca58832d5bfc061495d4aa8eeb",
        "dataset_version": "1",
        "file_name": "D04xian_led_0.jpg"
      }
    },

{
                      "GT": [
                          {
                              "bbox": [
                                  41.6,
                                  114.3,
                                  309.5,
                                  250.7
                              ],
                              "category_name": "car",
                              "category_probs": []
                          },
                          {
                              "bbox": [
                                  228.8,
                                  105.4,
                                  77.4,
                                  266.2
                              ],
                              "category_name": "person",
                              "category_probs": []
                          }
                      ],
                      "IFR": [
                          {
                              "bbox": [
                                  185,
                                  96,
                                  136,
                                  286
                              ],
                              "category_name": "person",
                              "category_probs": [
                                  0,
                                  0.8028634786605835
                              ]
                          },
                          {
                              "bbox": [
                                  228,
                                  130,
                                  75,
                                  210
                              ],
                              "category_name": "person",
                              "category_probs": [
                                  0,
                                  0.9659780859947205
                              ]
                          },
                          {
                              "bbox": [
                                  87,
                                  115,
                                  126,
                                  257
                              ],
                              "category_name": "car",
                              "category_probs": [
                                  0.32220494747161865,
                                  0
                              ]
                          }
                      ],
                      "file_name": "test1.png",
                      "file_path": "test/test1"
                  },
{
                      "GT": [
                          {
                              "bbox": [
                                  158.9,
                                  188.6,
                                  50,
                                  154
                              ],
                              "category_name": "car",
                              "category_probs": []
                          }
                      ],
                      "IFR": [
                          {
                              "bbox": [
                                  154,
                                  194,
                                  53,
                                  142
                              ],
                              "category_name": "person",
                              "category_probs": [
                                  0,
                                  0.834136962890625
                              ]
                          }
                      ],
                      "file_name": "test2.png",
                      "file_path": "test/test2"
                  }

  ]
}
metricHelper = ObjDetectionMetricHelper(input_json)
fa = metricHelper.getFA()
print(fa)