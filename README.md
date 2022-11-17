# code_ego4d

Este es el código que estoy usando para las modificaciones y las pruebas.

El clip_annotations está configurado para los videos que yo tengo descargados, puede haber alguna discrepancia con el que tengas descargado, si ese es el caso tienes que ejecutar el código con tu clip_annotations.

Para ejecutar el training usa yo uso:

python Train.py --use_xGPN --is_train true --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models --batch_size 11 --train_lr 0.0001 --num_epoch 30

El batch_size 11 es el máximo que se puede ejecutar con Peregrine porque es muy grande.

Una vez lo tienes entrenado puedes yo ejecuto lo siguente para generar las predicciones:

python Infer.py  --use_xGPN --is_train false --dataset ego4d --feature_path /data/s5091217/Ego4d-main/ego4d_data/v1/slowfast8x8_r101_k400 --checkpoint_path /data/s5091217/Ego4d-main/ego4d_data/v1/moments_models  --output_path /data/s5091217/Ego4d-main/ego4d_output --batch_size 11

He modificado el código para que por defecto se ejecute sobre el subconjunto de validación, no el de test porque no tiene las annotations publicadas aún.

Para evaluar los resultados de la predicción uso:

python Eval.py --dataset ego4d --output_path /data/s5091217/Ego4d-main/ego4d_output --out_prop_map true --eval_stage all

Aproximadamente toda esta ejecución tarda un poco menos de una hora.

Los parametros y resultados de cada ejecución se almacenan en el history.json (pero hay un problema, si cambias los parámetros y el training falla, el eval se ejecuta igual y guardara los resultados como si sí que se hubiera acabado el training pero usando el checkpoint del último training que se haya hecho.

También he añadido que se sincronice con wandb por defecto y vaya enviando la loss después de cada epoch, si no quieres esto añade --not_wandb al Train

Código no acabado:
- Hay código para ejecutarlo con ViT pero no está acabado
- ExecAll es un código experimental para ejecutar el train, infer y eval consecutivamente pero da un problema que creo que está relacionado con cómo se ejecutan los threads y no me he puesto a resolverlo

