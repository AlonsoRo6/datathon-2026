# RawTree Challenge · Recomanador en Temps Real
**Datathon FME 2026 · UPC Barcelona**  
*Equip 11 (`PepChavarriaLovers`)*

Aquest projecte és un sistema intel·ligent que recomana **10 productes** a cada comprador que entra a una botiga online. Tot funciona de forma **100% autònoma i en temps real**: cada pocs segons arriba una persona, mirem què ha estat fent a la botiga i, en **menys de 50 mil·lisegons**, li ensenyem els productes que té més números de comprar.

L'objectiu principal és **vendre més** que el sistema clàssic (que només mostrava els productes més venuts de sempre) i adaptar-nos als canvis com ara productes nous, canvis de tendència o productes que s'esgoten.

---

## 1. Com funciona el sistema (a vista d'ocell)

Per a no anar lents ni perdre cap client pel camí, hem separat la feina en dos fils que corren alhora:

```
                  ┌────────────────────────────────────────────────┐
                  │                 BOTIGA ONLINE                  │
                  │   (clients que arriben, clics, compres, estoc) │
                  └───────┬────────────────────────────────▲───────┘
  1. Arriben clients      │                                │ 4. Enviem els 10 productes
                          ▼                                │    en menys de 50 ms
                ┌──────────────────┐                       │
                │ BUCLE PRINCIPAL  │───────────────────────┘
                │ Llegir client    │
                │ Escollir 10 prods│
                └────────▲─────────┘
                         │ Consulta ràpida a la memòria
                         │ (dades sempre fresques i al dia)
                ┌────────┴─────────┐
                │ FIL EN SEGON PLA │
                │ Actualitza estoc │
                │ Detecta modes    │
                │ Mira compres     │
                └──────────────────┘
```

1. **El Bucle Principal (Molt ràpid):**
   * **Llegeix** els compradors que acaben d'arribar.
   * **Calcula** quins 10 productes encaixen millor amb cadascun.
   * **Envia** la resposta en bloc per no col·lapsar la xarxa. Tot aquest procés triga uns 25-50 mil·lisegons (tenim fins a 3 segons de marge).

2. **El Vigilant en Segon Pla (Feina de fons):**
   * Treballa sense destorbar el bucle principal.
   * Cada segon comprova si algun producte s'ha esgotat perquè no el recomanem mai per error.
   * Revisa què s'està venent a la botiga per detectar ràpidament quins productes s'han posat de moda.
   * Comprova què ha comprat cada usuari recentment per no recomanar-li una cosa que ja acaba d'adquirir.

---

## 2. Com triem els 10 productes? (Pas a pas)

Passar de 5.500 productes a només 10 per a cada persona es fa en tres passos senzills:

### Pas 1: La tria de candidats (d'on traiem les idees?)
No analitzem els 5.500 productes un per un perquè trigaríem massa. En lloc d'això, fem una preselecció ràpida d'uns 300-400 candidats interessants combinant diferents pistes:
* **El que està mirant ara mateix:** Si està mirant unes sabatilles, aquestes tenen moltes opcions de ser el que vol comprar.
* **Productes relacionats (Co-visites):** Què acostuma a mirar la gent després de veure aquest mateix producte?
* **Productes que es compren junts (Co-compres intel·ligents):** Mirem quines coses compra la gent juntes habitualment (per exemple, una càmera i la seva targeta de memòria), vigilant de no posar només els supervendes que compra tothom.
* **El millor de la seva subcategoria:** Si mira texans d'home, busquem els més venuts d'aquesta subcategoria concreta, i no pas roba en general.
* **Què està triomfant ara mateix (Tendències):** Productes que en l'última hora estan tenint un volum de visites i compres molt per sobre del que és habitual.

### Pas 2: Puntuar i ordenar amb Machine Learning
Un cop tenim aquests 300 candidats, els passem pel nostre model d'intel·ligència artificial (un *ensemble* basat en arbres de decisió: LightGBM i XGBoost). 

El model mira característiques com:
* **El context:** És des del mòbil o des de l'ordinador? A quina pàgina és (portada o fitxa de producte)? A quina hora?
* **El client:** Si ja el coneixem de dies anteriors, quines categories li agraden i quant acostuma a gastar-se.
* **El moment:** Quant de temps fa que mira productes en aquesta sessió?
* **La descripció del producte:** Analitzem el text de la descripció per veure com s'assembla al que està mirant.

El model posa una nota de 0 a 1 a cada candidat segons la probabilitat que sigui comprat.

### Pas 3: Muntar el carrusel de 10 (On col·loquem cada cosa?)
L'ordre en què col·loques els productes és clau, perquè **la gent mira sobretot els primers calaixos (d'esquerra a dreta)**:
* **Calaix 1:** Si l'usuari és a la fitxa d'un producte, sovint el posarem aquí. Si l'està mirant, és el seu màxim focus d'atenció.
* **Calaixos 2 al 4:** Els productes que han tret la millor nota del model (màxima compatibilitat).
* **Calaix 5:** Si el client és nou o el producte és acabat d'arribar, reservem aquest lloc per a donar visibilitat a novetats destacades.
* **Calaixos 6 al 10:** La resta de recomanacions de més qualitat del model.

---

## 3. Com resolem el "Cold Start" (Clients nous o productes acabats d'arribar)

Aquest és un dels reptes més difícils en qualsevol recomanador: què fas quan no tens dades prèvies?

### Cas A: Un client que no hem vist mai a la vida (`is_new_user`)
* **Si ja ha mirat algun producte a la sessió:** Ens oblidem del seu passat (que no existeix) i ens centrem 100% en el que està fent en aquest instant: la subcategoria concreta i el que la gent acostuma a mirar a continuació.
* **Si acaba d'entrar a la portada i no ha clicat res:** En lloc d'ensenyar-li 10 productes de tecnologia a cegues, fem un aparador variat amb el millor producte de cadascuna de les 10 categories. Així té una porta d'entrada sigui quin sigui el seu interès.

### Cas B: Un producte que s'acaba d'afegir al catàleg
Un producte acabat d'estrenar no té cap venda ni visita prèvia. Un model normal el descartaria per falta d'historial. Per solucionar-ho:
* **Busquem el seu "germà bessó":** Llegim la descripció del nou producte i busquem quin producte del catàleg s'hi assembla més en text i categoria. Mentre el nou producte no tingui dades pròpies, li "manllevem" les relacions al seu bessó.
* **Li donem un vot de confiança:** No li posem una nota de zero vendes; li assignem d'entrada el nivell de vendes habitual de la seva subcategoria perquè el model no el discrimini.
* **L'ensenyem al calaix 5:** Li garantim espai d'aparador en visites fredes perquè la gent tingui l'oportunitat de descobrir-lo i comprar-lo.

---

## 4. Com aprenem en directe durant la competició?

El sistema no està "congelat", sinó que aprèn contínuament mentre la botiga funciona:

1. **Control d'estoc instantani (cada 0.5 segons):** Si una talla o producte s'acaba, s'esborra de la llista al moment. Mai recomanem res que no es pugui comprar.
2. **Detecció de modes d'última hora:** Comparem què s'ha venut l'última hora amb el ritme de tot el dia. Si un article es comença a vendre de sobte a la ronda final, puja ràpidament al llistat de tendències.
3. **Afinitats d'avui en directe:** Cada minut revisem quines parelles de productes s'estan mirant juntes avui, adaptant-nos a canvis en el comportament dels compradors d'aquest matí.
4. **Auto-correcció segons les nostres vendes (Bandit):** Mirem quins productes estan comprant els clients a través de les nostres pròpies recomanacions. Si veiem que un article té molt d'èxit, li donem una petita bonificació a la seva nota perquè surti més sovint.

---

## 5. Estructura de fitxers del projecte

Per entendre on està cada part del codi:

* **`starter/recommender.py`:** El cervell d'execució. Conté el bucle que escolta RawTree, demana les recomanacions i les envia, a més de córrer les tasques de manteniment de fons.
* **`model/predictor.py`:** On es prenen les decisions. S'encarrega de triar els candidats, gestionar el *cold start*, calcular els bessons per text i ordenar el carrusel de 10.
* **`model/featurize.py`:** Converteix la informació del client i del producte en números que el model de Machine Learning pugui entendre.
* **`model/build_training.py`:** El programa que vam fer servir per entrenar els models d'arbres amb les dades dels 60 dies històrics.
* **`model/out/`:** On estan guardats els models entrenats (`model_lgbm_v3.txt`, `model_xgb_v3.json`) i la seva configuració.
* **`features/`:** Taules auxiliars preparades (co-visites històriques, productes comprats junts, diccionaris de paraules).
* **`run_live.sh`:** L'script per deixar el sistema funcionant en viu de manera contínua, evitant que l'ordinador s'adormi i reiniciant el procés si hi ha cap incidència.

---

## 6. Rendiment assolit

* **Velocitat:** Cada resposta es calcula en **~25 mil·lisegons**, molt per sota del màxim de 3 segons permès.
* **Puntualitat:** Més del **99.2%** de les respostes s'han lliurat a temps i de forma vàlida.
* **Resultats de Venda:**
  * El sistema bàsic de la botiga venia a un ritme de **~0.53%** de conversions.
  * El nostre sistema supera el **1.34%** de conversió global (més de $2.5\times$ les vendes del sistema clàssic) i un **1.68%** en casos de *cold start*.

---

## 7. Com engegar-lo

1. Tenir un fitxer `.env` amb les credencials del repte (`TEAM_ID`, `RAWTREE_API_KEY`).
2. Instal·lar els paquets necessaris:
   ```bash
   pip install -r starter/requirements.txt
   ```
3. Posar en marxa el sistema amb el supervisor automàtic:
   ```bash
   bash run_live.sh
   ```
   *(També es pot executar directament amb `python starter/recommender.py`)*.
4. Seguir l'activitat en directe:
   ```bash
   tail -f run.log
   ```
