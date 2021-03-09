"""Helper script for deleting the ignore regions from BDD100K COCO format
annotations"""

import os
import json
import argparse


def main(args: argparse.Namespace) -> None:
    """Main function. Delete ignore regions and save updated annotation
     files."""

    with open(args.annotation_file) as f:
        ori_coco = json.load(f)

    ori_annos = ori_coco['annotations']
    new_annos = []
    for ori_anno in ori_annos:
        if ori_anno['category_id'] > 10:
            continue
        new_annos.append(ori_anno)

    new_coco = ori_coco
    new_coco['annotations'] = new_annos
    new_coco['categories'] = ori_coco['categories'][:-1]

    os.makedirs(args.save_dir, exist_ok=True)
    new_fn = os.path.join(args.save_dir,
                          os.path.basename(args.annotation_file))
    with open(new_fn, 'w') as f:
        json.dump(new_coco, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Delete ignore regions in BDD100K coco format annotations '
                    '(not supported in detectron2)')
    parser.add_argument('-a', '--annotation-file',
                        help='path to coco format annotation file for BDD100K')
    parser.add_argument('-s', '--save-dir',
                        help='save dir for new coco format annotation file')
    main(parser.parse_args())

    # TODO move to BDD100k --> label
