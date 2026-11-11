**Objetivo:** Entender qué son los embeddings y cómo se compara la similitud entre textos

**Material académico:**
[C1, 00:18:32.360–00:18:50.210] Marina Fuster: Esto ya lo fuimos viendo con la motivación cómo representar el significado de una palabra. Voy a usar palabras como reemplazo de unidad lingüística, Puede ser palabra, documento, su palabra, etcétera, etcétera. Me interesan un montón de cosas que caen bajo un campo que se llama lexical Semántics.
[C2, 00:20:32.610–00:21:00.349] Marina Fuster: Y el chiste era que si yo veía estos vectores en el espacio.
Marina Fuster: el ángulo entre esos vectores. Hablamos de esto como medida de similitud. Me podría decir que
Marina Fuster: qué palabras por ahí eran parecidas o no. Fíjense acá. Que digital e information las toman como muy parecidas, porque el ángulo es muy chiquito, a pesar de que information aparece muchísimas más veces Por eso la magnitud es más grande.
[C3, 00:25:22.170–00:25:37.590] Marina Fuster: Entonces ahí es donde aparece el tema de los Emedings, que son vectores dimensionales y que ellos suele asociarse a que son vectores densos. No, yo tengo como una especie de representación más comprimida que estas matrices. Es falsas, que son las primeras implementaciones. Si.
[C4, 00:25:37.950–00:27:13.979] Marina Fuster: como mido, la similitud puede usar una métrica como similitud coseno, es la única, no tiene sus ventajas, pero también tiene sus defectos. Por ejemplo, si a mí me importa la magnitud del vector porque me representa algo importante: el medir la similitud coseno me aplana esa
Marina Fuster: esa importancia. Por ejemplo, si yo tengo, bueno.
Marina Fuster: alguien que mira series, y en la X tengo series románticas
Marina Fuster: y en la I tengo series de terror. Soy, No conozco ninguna serie de terror. Bueno, la persona que mire 100 románticas y 100 de terror va a tener la misma similitud con 0 que la que mira 1 con una, no.
Marina Fuster: ¿por qué el vector apunta. Pues así, entonces, Fíjense que en este caso yo quizás estoy tratando como iguales 2 puntos en el espacio que son sumamente distintos. Uno es adicto a ver series, y el otro vio algo un poco de cada cosa. Si entonces
Marina Fuster: similitud con seno, se puede usar. Fíjense que en la otra en realidad está. Bueno, porque si yo tengo, no sé una palabra que no se usa mucho como escarlata y
Marina Fuster: rojo rojo se usó un montón escarlata, ¿no? Entonces ahí quizás la frecuencia no me importa tanto escarlata. Es un sinónimo de rojo, entonces tienen sus ventajas y desventajas depende de donde las uso, pero nos vamos a conformar con quedarnos con la similitud con 0. Por ahora.
[C5, 00:27:47.050–00:28:44.829] Marina Fuster: el estático. La característica principal es que mantiene el valor fijo. O sea, yo tengo una instancia de entrenamiento y de aprender los emails, pero una vez que terminé.
Marina Fuster: El emending va a ser siempre el mismo para la palabra. Si, por ejemplo.
Marina Fuster: yo tengo un algoritmo de estática en Beit, si yo tengo estas 2 frases
Marina Fuster: que comí pizza o que la pizza era asquerosa. El embelle para pizza va a ser el mismo. Se va a posicionar ahí? Sí,
Marina Fuster: En cambio, los dinámicos van a cambiar de valor dependiendo del contexto en esta en la instancia particular, entonces el ejemplo sería algo así. No supongamos que el eje X se relaciona con qué tan rico es la comida.
Marina Fuster: Si yo tengo la frase que la pizza era asquerosa. Bueno, yo esperaría que el impending no sea el mismo que
Marina Fuster: la pizza común, que se asocia como que es más rica.
[C6, 00:31:55.350–00:32:14.820] Marina Fuster: esquemáticamente lo que empieza a pasar, porque fíjense que el el dot product que yo hago
