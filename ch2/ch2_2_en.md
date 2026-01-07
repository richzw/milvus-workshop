## 2.2 Data Insertion and Management [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richzw/milvus-workshop/blob/main/ch2/ch2_2_en.ipynb) 

In the previous section, we learned how to create and manage Collections. Now, we will learn how to insert data into these Collections and perform basic management operations such as deletion.

### Concept: Entity

In Milvus, an **Entity** is the fundamental unit of data, representing a single object or record stored within a Collection.

- **Structure**: Each entity contains a set of fields, whose definitions adhere to the schema of its parent Collection.
- **Composition**: An entity must include at least one primary key field and one or more vector fields. It typically also contains additional scalar fields for metadata or filtering conditions.
- **Analogy**: If a Collection is likened to a "table" in a database, then an entity corresponds to a "row of data" within that table or a "document" in a NoSQL database.
- **Uniqueness**: Each entity is uniquely identified by its primary key.

### Concept: Partition

**Partition** is an optional data partitioning mechanism within a Collection. It allows you to split a large Collection into multiple smaller, more manageable parts.

- **Purpose**:
    - **Improve Search Efficiency**: During searches, you can specify to search within one or more specific partitions, thereby narrowing the search scope and accelerating query speeds.
    - **Data Management**: You can perform Load, Release, or Drop operations on specific partitions, facilitating lifecycle management for different datasets. For example, create partitions by date, category, etc.
    - **Data Isolation**: Data within different partitions can be physically organized more compactly.
- **Features**:
    - Each Collection can contain multiple partitions.
    - Each partition has a unique name.
    - If partitions are not created, all data is stored by default in a partition named `_default`.
    - An entity can belong to only one partition.
- **Operations**: Create partitions, delete partitions, list partitions, check partition existence, etc.

### Hands-on: Preparing Data for Insertion

Before inserting data into Milvus, we must prepare it according to the Collection's Schema. Data is typically organized as Python lists, where each list element can be either a dictionary (recommended for better readability) or a tuple (field order must strictly match the Schema).

We will prepare some sample data for the `book_search` Collection created in Hands-on Exercise 1 (or recreate it if it doesn't exist).

Schema for `book_search`:
1.  `book_id`: `INT64`, primary key, auto-generated ID
2.  `book_title`: `VARCHAR`, max_length=512
3.  `publication_year`: `INT32`
4.  `book_embedding`: `FLOAT_VECTOR`, dim=768


```python
!pip install numpy==1.26.4
```


```python
# Import necessary libraries
import random
import numpy as np
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

MILVUS_URI = "http://localhost:19530"
client = MilvusClient(uri=MILVUS_URI)

# Define Collection name
EXERCISE_COLLECTION_NAME = "book_search"

# Check and recreate Collection
if client.has_collection(collection_name=EXERCISE_COLLECTION_NAME):
    print(f"The existing Collection '{EXERCISE_COLLECTION_NAME}' has been detected and will be deleted.")
    client.drop_collection(collection_name=EXERCISE_COLLECTION_NAME)

field_book_id = FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
field_book_title = FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=512)
field_publication_year = FieldSchema(name="publication_year", dtype=DataType.INT32)
field_book_embedding = FieldSchema(name="book_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
book_schema_def = CollectionSchema(
    fields=[field_book_id, field_book_title, field_publication_year, field_book_embedding],
    description="Collection for storing book information and embeddings (MilvusClient)",
    enable_dynamic_field=False
)
client.create_collection(
    collection_name=EXERCISE_COLLECTION_NAME,
    schema=book_schema_def,
    consistency_level="Strong"
)
print(f"Collection '{EXERCISE_COLLECTION_NAME}' has been created.")

```

    The existing Collection 'book_search' has been detected and will be deleted.
    Collection 'book_search' has been created.



```python
# Prepare simulated data
NUM_ENTITIES = 100
DIMENSION = 768 # Must align with the dimensions defined in the Collection Schema

# Generate simulated data
data_to_insert = []
for i in range(NUM_ENTITIES):
    entity = {
        "book_title": f"The Amazing Book Title {i+1}",
        "publication_year": random.randint(1980, 2023),
        "book_embedding": np.random.rand(DIMENSION).astype(np.float32).tolist() # Generate random vectors
    }
    data_to_insert.append(entity)

print(f"Successfully generated {len(data_to_insert)} simulated records.")
print("First data example:")
print(data_to_insert[0])
```

    Successfully generated 100 simulated records.
    First data example:
    {'book_title': 'The Amazing Book Title 1', 'publication_year': 2007, 'book_embedding': [0.7554160356521606, 0.2608287036418915, 0.947449803352356, 0.3069806694984436, 0.9613373875617981, 0.3784532845020294, 0.43865251541137695, 0.1481749266386032, 0.02266550622880459, 0.5453378558158875, 0.2668735682964325, 0.0010517380433157086, 0.9731225371360779, 0.3236008584499359, 0.12104277312755585, 0.25444766879081726, 0.43917471170425415, 0.7247389554977417, 0.3850972056388855, 0.9846354722976685, 0.48755019903182983, 0.0011725855292752385, 0.5226274132728577, 0.526183545589447, 0.9840882420539856, 0.93752121925354, 0.35475555062294006, 0.724798321723938, 0.7902653217315674, 0.9286869764328003, 0.7599101662635803, 0.6498773694038391, 0.030379414558410645, 0.25534525513648987, 0.5476850867271423, 0.2174743413925171, 0.2525869607925415, 0.4075888991355896, 0.6774653196334839, 0.35165050625801086, 0.811332643032074, 0.1696503758430481, 0.4743022620677948, 0.03257852420210838, 0.11186164617538452, 0.009938117116689682, 0.14515632390975952, 0.22805556654930115, 0.4900100529193878, 0.15294982492923737, 0.9463433027267456, 0.7299997806549072, 0.5387651324272156, 0.46339964866638184, 0.39222389459609985, 0.9254282116889954, 0.8160644769668579, 0.2645498514175415, 0.5309605002403259, 0.1464931070804596, 0.413151353597641, 0.04087236896157265, 0.667694091796875, 0.7405688166618347, 0.07079742103815079, 0.898158848285675, 0.9904046654701233, 0.8637321591377258, 0.008836123161017895, 0.681088924407959, 0.5771447420120239, 0.8763667345046997, 0.2874665856361389, 0.6783254146575928, 0.12035103887319565, 0.8120216727256775, 0.7300283908843994, 0.7797582149505615, 0.9339454770088196, 0.9306850433349609, 0.054972145706415176, 0.03206039220094681, 0.5183939933776855, 0.046043407171964645, 0.1327282041311264, 0.6977534890174866, 0.5551252961158752, 0.27098095417022705, 0.6332252621650696, 0.9961817264556885, 0.1974768489599228, 0.48585841059684753, 0.20090585947036743, 0.9014126062393188, 0.0009869972709566355, 0.1868332028388977, 0.6470547914505005, 0.7845665812492371, 0.8186453580856323, 0.9057676196098328, 0.6769302487373352, 0.7985094785690308, 0.3653392791748047, 0.3645249009132385, 0.41137564182281494, 0.1391855925321579, 0.44895046949386597, 0.41734445095062256, 0.7164074778556824, 0.22909820079803467, 0.22404389083385468, 0.43668603897094727, 0.09576534479856491, 0.758048415184021, 0.15666337311267853, 0.6441246867179871, 0.9548223614692688, 0.8184003829956055, 0.4649474620819092, 0.9414393901824951, 0.3950771689414978, 0.32158172130584717, 0.11848891526460648, 0.9751256704330444, 0.09162876754999161, 0.15820017457008362, 0.399566650390625, 0.54273921251297, 0.2024572491645813, 0.3217657208442688, 0.3814920485019684, 0.5677749514579773, 0.465892493724823, 0.24519358575344086, 0.9068989753723145, 0.7417060136795044, 0.22099755704402924, 0.87442547082901, 0.8509578108787537, 0.661394476890564, 0.15219803154468536, 0.0013111232547089458, 0.47554323077201843, 0.139320969581604, 0.25243261456489563, 0.9835110902786255, 0.47583311796188354, 0.24669945240020752, 0.8883482813835144, 0.9907375574111938, 0.18118172883987427, 0.6511966586112976, 0.5749350786209106, 0.824579119682312, 0.23431292176246643, 0.5851580500602722, 0.6526411771774292, 0.7719447016716003, 0.8753591179847717, 0.37553170323371887, 0.5877018570899963, 0.08547880500555038, 0.9627142548561096, 0.9019086360931396, 0.8830283880233765, 0.42577874660491943, 0.8428781032562256, 0.9259962439537048, 0.1524343639612198, 0.11487050354480743, 0.3446078598499298, 0.16475896537303925, 0.318347305059433, 0.14699645340442657, 0.7631383538246155, 0.8876062631607056, 0.8218327760696411, 0.7435257434844971, 0.3598456084728241, 0.23449058830738068, 0.8978416323661804, 0.9398496747016907, 0.11321984231472015, 0.5371344089508057, 0.16468371450901031, 0.3666720986366272, 0.905834972858429, 0.9245976209640503, 0.9315714836120605, 0.7947924137115479, 0.9877877235412598, 0.6160137057304382, 0.8950866460800171, 0.7606651186943054, 0.9740989208221436, 0.8308684825897217, 0.15169250965118408, 0.6113812327384949, 0.48632559180259705, 0.06463273614645004, 0.9289991855621338, 0.274965763092041, 0.2962639033794403, 0.003726436523720622, 0.8798927068710327, 0.7393850684165955, 0.5271127820014954, 0.37764284014701843, 0.577401876449585, 0.8914909958839417, 0.5240945219993591, 0.16614213585853577, 0.9245126843452454, 0.7070824503898621, 0.216993510723114, 0.7267552018165588, 0.2673121690750122, 0.5606276988983154, 0.9846664071083069, 0.972504734992981, 0.13018836081027985, 0.6948012709617615, 0.24201761186122894, 0.7432515025138855, 0.9347618818283081, 0.5260328054428101, 0.41482868790626526, 0.5793872475624084, 0.7724622488021851, 0.3016027808189392, 0.8097487092018127, 0.23858259618282318, 0.013798931613564491, 0.009901043027639389, 0.7537766098976135, 0.6694875359535217, 0.39774495363235474, 0.858959972858429, 0.33907559514045715, 0.07360665500164032, 0.7389338612556458, 0.3027888536453247, 0.8634902238845825, 0.5240959525108337, 0.38747626543045044, 0.8892226219177246, 0.621944785118103, 0.005502572748810053, 0.6627065539360046, 0.40873339772224426, 0.2744508981704712, 0.5893881320953369, 0.13919752836227417, 0.11358390003442764, 0.7485488057136536, 0.5069100856781006, 0.34401535987854004, 0.5056793689727783, 0.6378872394561768, 0.029355565086007118, 0.716014564037323, 0.9850488305091858, 0.10197513550519943, 0.46816280484199524, 0.7205294966697693, 0.024869274348020554, 0.4414180517196655, 0.03662775829434395, 0.8642598390579224, 0.0311715230345726, 0.48412981629371643, 0.12733078002929688, 0.021508989855647087, 0.15081115067005157, 0.40324240922927856, 0.7773968577384949, 0.843693196773529, 0.7711021304130554, 0.37405046820640564, 0.2161625474691391, 0.14599765837192535, 0.2282579243183136, 0.5395952463150024, 0.983274519443512, 0.169753760099411, 0.37703800201416016, 0.07941056787967682, 0.09355700016021729, 0.6121503114700317, 0.44010528922080994, 0.41567564010620117, 0.19005325436592102, 0.18968743085861206, 0.2361917793750763, 0.366199791431427, 0.026276057586073875, 0.13890206813812256, 0.31375330686569214, 0.11610841751098633, 0.8698614239692688, 0.37528467178344727, 0.008839219808578491, 0.5821777582168579, 0.29295971989631653, 0.7549133896827698, 0.8950487375259399, 0.8759481310844421, 0.9720003604888916, 0.12087024003267288, 0.8915160298347473, 0.30210691690444946, 0.22886928915977478, 0.5827834010124207, 0.6615564227104187, 0.9563537240028381, 0.07294991612434387, 0.2566065192222595, 0.33366796374320984, 0.2931543290615082, 0.2748214602470398, 0.5662650465965271, 0.6611282825469971, 0.9336940050125122, 0.4847327470779419, 0.2980804145336151, 0.06616046279668808, 0.31540027260780334, 0.2629932761192322, 0.2675536274909973, 0.4228147864341736, 0.7742351293563843, 0.22351977229118347, 0.30712029337882996, 0.5898292660713196, 0.6751102209091187, 0.18984505534172058, 0.2954311966896057, 0.7069984078407288, 0.07602953910827637, 0.23728518187999725, 0.7538627982139587, 0.24571585655212402, 0.24197077751159668, 0.5188281536102295, 0.7071848511695862, 0.534731924533844, 0.2651609480381012, 0.2839130759239197, 0.32531651854515076, 0.7831250429153442, 0.8852272629737854, 0.8648989796638489, 0.45574894547462463, 0.8937836289405823, 0.6863552927970886, 0.2582421600818634, 0.3822422921657562, 0.836458146572113, 0.2120387703180313, 0.4989001452922821, 0.8703686594963074, 0.7395424246788025, 0.7798870801925659, 0.2227000594139099, 0.6143176555633545, 0.8701836466789246, 0.25055351853370667, 0.613958477973938, 0.9674338698387146, 0.472468763589859, 0.5758557915687561, 0.7937012314796448, 0.5543349981307983, 0.9742078185081482, 0.3466459810733795, 0.8077684640884399, 0.45970889925956726, 0.5384522080421448, 0.7467844486236572, 0.6542564630508423, 0.6415061950683594, 0.7027416229248047, 0.7314454317092896, 0.9183827638626099, 0.4371454417705536, 0.5670241117477417, 0.2875922918319702, 0.17666387557983398, 0.17784401774406433, 0.1483885943889618, 0.7348068356513977, 0.7737049460411072, 0.47895023226737976, 0.0188915878534317, 0.767593502998352, 0.3390744924545288, 0.4383287727832794, 0.7837225198745728, 0.5013964772224426, 0.5825387239456177, 0.0870150476694107, 0.6017679572105408, 0.2697373032569885, 0.7372795939445496, 0.40812331438064575, 0.023717131465673447, 0.7004799842834473, 0.9445147514343262, 0.03288945555686951, 0.47238266468048096, 0.9929032325744629, 0.9997323751449585, 0.5014985799789429, 0.34926849603652954, 0.2576070725917816, 0.26062750816345215, 0.8679243922233582, 0.14423637092113495, 0.7375307083129883, 0.08624446392059326, 0.35996657609939575, 0.11030241847038269, 0.050762563943862915, 0.9951047897338867, 0.009364494122564793, 0.03929569944739342, 0.32924190163612366, 0.018559373915195465, 0.17287498712539673, 0.30809107422828674, 0.29544007778167725, 0.295564204454422, 0.49905863404273987, 0.3543567955493927, 0.6978399157524109, 0.9959014654159546, 0.03303297981619835, 0.14289820194244385, 0.33239084482192993, 0.624527633190155, 0.09335947781801224, 0.5669840574264526, 0.4710410237312317, 0.1810307502746582, 0.4415164291858673, 0.9431402683258057, 0.8269506692886353, 0.033377066254615784, 0.7615782618522644, 0.569636881351471, 0.5161347389221191, 0.15375195443630219, 0.12595364451408386, 0.03231999650597572, 0.7385687232017517, 0.8625124096870422, 0.86992347240448, 0.1596219688653946, 0.18570782244205475, 0.33241698145866394, 0.44680821895599365, 0.7697407603263855, 0.2797175943851471, 0.26129892468452454, 0.412309855222702, 0.10280391573905945, 0.009394018910825253, 0.5072442889213562, 0.7301130294799805, 0.8616156578063965, 0.5417841672897339, 0.2988694906234741, 0.48915785551071167, 0.8518503308296204, 0.4859418570995331, 0.11314260214567184, 0.6592094302177429, 0.9497263431549072, 0.2857714295387268, 0.8384438157081604, 0.7914565205574036, 0.3633436858654022, 0.042389512062072754, 0.19832907617092133, 0.1304231584072113, 0.43598777055740356, 0.32085537910461426, 0.09477058798074722, 0.2771519124507904, 0.5998687744140625, 0.6740500330924988, 0.8758850693702698, 0.08209959417581558, 0.9555155038833618, 0.6815202236175537, 0.15205252170562744, 0.9502594470977783, 0.006613962817937136, 0.7535254955291748, 0.14676739275455475, 0.6193066239356995, 0.01844526082277298, 0.5393730401992798, 0.324411541223526, 0.5338190793991089, 0.13022492825984955, 0.7621189951896667, 0.6492626070976257, 0.16485409438610077, 0.6218110918998718, 0.7637635469436646, 0.387739896774292, 0.7870835065841675, 0.7720727920532227, 0.7812526822090149, 0.07958370447158813, 0.5556883811950684, 0.22761254012584686, 0.24475210905075073, 0.1494504064321518, 0.8982670307159424, 0.8416599631309509, 0.3565642833709717, 0.9581728577613831, 0.9294024705886841, 0.17292456328868866, 0.5365901589393616, 0.7468444108963013, 0.6359759569168091, 0.7950950264930725, 0.800302267074585, 0.8260029554367065, 0.08810830861330032, 0.034726228564977646, 0.6145686507225037, 0.5759243369102478, 0.2770688533782959, 0.9793515801429749, 0.010171550326049328, 0.9370872974395752, 0.766828179359436, 0.7257584929466248, 0.16608668863773346, 0.12740258872509003, 0.5678361058235168, 0.5202620625495911, 0.1613946408033371, 0.06350576877593994, 0.9965569376945496, 0.5094914436340332, 0.8657680749893188, 0.17648278176784515, 0.9705206751823425, 0.6389328837394714, 0.3993641138076782, 0.8657519221305847, 0.6091623306274414, 0.48169171810150146, 0.799258828163147, 0.8295344114303589, 0.7066593170166016, 0.4141719937324524, 0.05054814741015434, 0.02283203788101673, 0.004575933329761028, 0.0725482627749443, 0.782482922077179, 0.7222043871879578, 0.5561830997467041, 0.07468281686306, 0.48529836535453796, 0.8593477606773376, 0.8539935350418091, 0.4975817799568176, 0.08247971534729004, 0.08915320038795471, 0.029608488082885742, 0.4627738893032074, 0.61183100938797, 0.6681979894638062, 0.7736825346946716, 0.023451825603842735, 0.7359476089477539, 0.0928163081407547, 0.9010948538780212, 0.8438407778739929, 0.5584729909896851, 0.19616544246673584, 0.517859935760498, 0.7371347546577454, 0.2741602659225464, 0.4568127691745758, 0.9237081408500671, 0.5432027578353882, 0.854256272315979, 0.24553252756595612, 0.8410641551017761, 0.4265466630458832, 0.6365063190460205, 0.4842962920665741, 0.9633913040161133, 0.9508780837059021, 0.8758758902549744, 0.3149005174636841, 0.458574116230011, 0.9108802080154419, 0.29435649514198303, 0.7824134826660156, 0.3068040907382965, 0.2997891902923584, 0.18480849266052246, 0.8947428464889526, 0.2150982767343521, 0.3687182068824768, 0.9752876162528992, 0.9830842614173889, 0.6804925799369812, 0.8902236223220825, 0.1598111391067505, 0.7453922033309937, 0.7994822263717651, 0.23408591747283936, 0.5844670534133911, 0.8512769937515259, 0.6074814200401306, 0.25267407298088074, 0.36911681294441223, 0.36103522777557373, 0.09150350093841553, 0.003359766211360693, 0.8712501525878906, 0.5191840529441833, 0.30424249172210693, 0.9345712661743164, 0.2580098509788513, 0.7066293954849243, 0.25041478872299194, 0.5996718406677246, 0.20047788321971893, 0.9297193884849548, 0.35332101583480835, 0.7711384892463684, 0.9291272163391113, 0.9829116463661194, 0.9159162044525146, 0.4101644456386566, 0.6154218316078186, 0.9004846215248108, 0.15623150765895844, 0.08855441212654114, 0.24726513028144836, 0.10425745695829391, 0.07577911764383316, 0.9455416798591614, 0.6578969359397888, 0.3615518808364868, 0.41546663641929626, 0.9618772864341736, 0.6706806421279907, 0.7138648629188538, 0.18077923357486725, 0.2512779235839844, 0.345491498708725, 0.6018428802490234, 0.64078688621521, 0.17077873647212982, 0.39343640208244324, 0.7062719464302063, 0.3306984603404999, 0.6474603414535522, 0.21169236302375793, 0.43974196910858154, 0.06259015202522278, 0.8710783123970032, 0.6682480573654175, 0.9797359108924866, 0.6223967671394348, 0.12634174525737762, 0.4089619815349579, 0.38417112827301025, 0.8420875668525696, 0.824715256690979, 0.7333439588546753, 0.5218288898468018, 0.5841250419616699, 0.4915536046028137, 0.1333373785018921, 0.8084954023361206, 0.592771053314209, 0.18497107923030853, 0.2370721697807312, 0.04208620265126228, 0.8309365510940552, 0.1810533106327057, 0.47406551241874695, 0.5459632277488708, 0.9709461331367493, 0.26226314902305603, 0.505805253982544, 0.2407677173614502, 0.8617594242095947, 0.8297126293182373, 0.9548647999763489, 0.3841170370578766, 0.28158167004585266, 0.9418792128562927, 0.5609622597694397, 0.2462398260831833, 0.9561352133750916, 0.31035593152046204, 0.25351589918136597, 0.9112070798873901, 0.7805351614952087, 0.6662537455558777, 0.18631578981876373, 0.18634478747844696, 0.08343230187892914, 0.7572841644287109, 0.26470449566841125, 0.09430951625108719, 0.7896186709403992, 0.7612367272377014, 0.29364001750946045, 0.5818741917610168, 0.926804780960083, 0.8573131561279297, 0.30962204933166504, 0.4914816915988922, 0.9129489064216614, 0.2866550385951996, 0.6281821131706238, 0.013710015453398228, 0.07828482985496521, 0.5044150352478027, 0.9513916373252869, 0.8710907697677612, 0.25438931584358215, 0.6212169528007507, 0.09225710481405258, 0.3746756911277771, 0.1569872498512268, 0.6200754046440125, 0.6571910381317139, 0.10097558051347733, 0.8227220773696899, 0.9296902418136597, 0.2546578049659729, 0.6015764474868774, 0.9381170272827148, 0.7319120764732361, 0.7489774227142334, 0.7216432094573975, 0.5455694794654846, 0.6593194007873535, 0.6969848275184631, 0.1831894963979721, 0.27983036637306213, 0.3798833191394806, 0.9648955464363098, 0.19815614819526672, 0.4487268328666687, 0.3577061593532562, 0.0916174054145813, 0.6536416411399841, 0.5545380115509033, 0.08807837218046188, 0.29715272784233093, 0.05643739551305771]}


### Hands-On: Creating and Managing Partitions (Optional)

While we can insert data directly into the collection's default partition (`_default`), we'll first demonstrate how to create and manage partitions.


```python
PARTITION_NAME_FICTION = "fiction_books"
PARTITION_NAME_NON_FICTION = "non_fiction_books"

# 1. Create Partitions
try:
    if not client.has_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_FICTION):
        client.create_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_FICTION)
        print(f"The partition '{PARTITION_NAME_FICTION}' was successfully created in the Collection '{EXERCISE_COLLECTION_NAME}'.")
    else:
        print(f"The partition '{PARTITION_NAME_FICTION}' already exists.")

    if not client.has_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_NON_FICTION):
        client.create_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_NON_FICTION)
        print(f"The partition '{PARTITION_NAME_NON_FICTION}' was successfully created in the Collection '{EXERCISE_COLLECTION_NAME}'.")
    else:
        print(f"The partition '{PARTITION_NAME_NON_FICTION}' already exists.")
except Exception as e:
    print(f"Failed to create partition: {e}")

# 2. List all partitions
try:
    partitions = client.list_partitions(collection_name=EXERCISE_COLLECTION_NAME)
    print(f"\nPartitions in Collection '{EXERCISE_COLLECTION_NAME}': {partitions}")
except Exception as e:
    print(f"List partitions failed: {e}")

# 3. Check if the partition exists
try:
    has_fiction = client.has_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_FICTION)
    print(f"Does the partition '{PARTITION_NAME_FICTION}' exist: {has_fiction}")
    has_scifi = client.has_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name="sci_fi_books") # A non-existent partition
    print(f"Does the partition 'sci_fi_books' exist: {has_scifi}")
except Exception as e:
    print(f"Check for partition failures: {e}")

# 4. Delete partition (if necessary, typically cleaned up after testing)
# try:
#     if client.has_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_NON_FICTION):
#         client.drop_partition(collection_name=EXERCISE_COLLECTION_NAME, partition_name=PARTITION_NAME_NON_FICTION)
#         print(f"The partition '{PARTITION_NAME_NON_FICTION}' has been deleted.")
# except Exception as e:
#     print(f"Delete partition failed: {e}")
```

    The partition 'fiction_books' was successfully created in the Collection 'book_search'.
    The partition 'non_fiction_books' was successfully created in the Collection 'book_search'.
    
    Partitions in Collection 'book_search': ['_default', 'fiction_books', 'non_fiction_books']
    Does the partition 'fiction_books' exist: True
    Does the partition 'sci_fi_books' exist: False


### Hands-On: Inserting Data into a Collection (or Specified Partition)

Use the `client.insert()` method to insert data into a specified Collection.

- **Data Format Requirements**:
    - `data`: A list where each element represents an entity.
         - For dictionary lists (recommended): `[{“field1”: value1, “field2”: value2, ...}, ...]`. Field names must match those in the Schema. For auto-generated ID primary keys, **do not** include the primary key field in the data.
         - For a list of tuples (or list of lists): `[(value_field1, value_field2, ...), ...]`. The order of values must strictly follow the sequence defined in `CollectionSchema.fields`. Similarly, the auto-generated ID primary key field should not contain a value.
 - **`partition_name` (optional)**: Specify this parameter to insert data into a specific partition. If omitted, data will be inserted into the `_default` partition.
- **Return Value**: The `insert()` method returns a `MutationResult` object containing `insert_count` (number of successfully inserted entities) and `primary_keys` (list of primary keys for newly inserted entities, especially important for auto-ID).

 - **Tips for Batch Insertion**:
    - `client.insert()` inherently supports batch insertion (by passing a list containing multiple entities).
    - Inserting a large batch at once is generally more efficient than inserting single entities in loops, as it reduces network communication overhead.
     - Milvus imposes limits on the data volume per insertion (typically constrained by gRPC message size limits, approximately 32MB-64MB depending on Milvus version and configuration). For extremely large datasets, batching at the application level is required. PyMilvus also performs some internal batch processing.


```python
# Insert the first half of the data into the 'fiction_books' partition, and insert the second half into the default partition.
num_to_fiction = NUM_ENTITIES // 2
data_for_fiction = data_to_insert[:num_to_fiction]
data_for_default = data_to_insert[num_to_fiction:]

inserted_pks = [] # Primary key used to store all inserted data, facilitating subsequent deletion operations.

# 1. Insert into the specified partition 'fiction_books'
try:
    print(f"\nPreparing to insert {len(data_for_fiction)} records into partition '{PARTITION_NAME_FICTION}'...")
    res_fiction = client.insert(
        collection_name=EXERCISE_COLLECTION_NAME,
        data=data_for_fiction,
        partition_name=PARTITION_NAME_FICTION
    )
    print(f"Successfully inserted {res_fiction['insert_count']} records into partition '{PARTITION_NAME_FICTION}'.")
    print(f"Primary keys returned (first 5): {res_fiction['ids'][:5]}")
    inserted_pks.extend(res_fiction['ids'])
except Exception as e:
    print(f"Failed to insert data into partition '{PARTITION_NAME_FICTION}': {e}")

# 2. Insert into the default partition (_default)
try:
    print(f"\nPreparing to insert {len(data_for_default)} records into the default partition...")
    res_default = client.insert(
        collection_name=EXERCISE_COLLECTION_NAME,
        data=data_for_default
        # If partition_name is omitted, insert into _default
    )
    print(f"Successfully inserted {res_default['insert_count']} records into the default partition.")
    print(f"Primary keys returned (first 5): {res_default['ids'][:5]}")
    inserted_pks.extend(res_default['ids'])
except Exception as e:
    print(f"Failed to insert data into the default partition: {e}")

```

    
    Preparing to insert 50 records into partition 'fiction_books'...
    Successfully inserted 50 records into partition 'fiction_books'.
    Primary keys returned (first 5): [461486305505040590, 461486305505040591, 461486305505040592, 461486305505040593, 461486305505040594]
    
    Preparing to insert 50 records into the default partition...
    Successfully inserted 50 records into the default partition.
    Primary keys returned (first 5): [461486305505040641, 461486305505040642, 461486305505040643, 461486305505040644, 461486305505040645]


 **Important: Flushing Data**

After inserting data, it is initially stored in memory buffers. To ensure data is persisted to disk and can be correctly processed by subsequent operations (such as index building or searches), you must execute the `flush()` operation.
`flush()` flushes the data segments in memory (growing segments) to disk, creating persistent data segments (sealed segments).

 Although Milvus has an automatic flush mechanism, it is good practice to manually call `client.flush()` before performing critical operations—such as building indexes, expecting immediate changes to `num_entities` after bulk deletions, or ensuring data is fully persisted.


```python
try:
    print(f"\nFlushing Collection '{EXERCISE_COLLECTION_NAME}'...")
    client.flush(collection_name=EXERCISE_COLLECTION_NAME) # Milvus 2.3+
    # For PyMilvus versions < 2.3.2 (roughly), if the client is created via MilvusClient,
    # flush may need to be performed using utility.flush([EXERCISE_COLLECTION_NAME]) or the Collection object.
    # However, MilvusClient 2.3+ should support direct client.flush().
    print("The flush operation has been requested. This may take some time to complete.")
    
    # Check entity count (num_entities should update after flush)
    # Note: Updates to num_entities may not be instantaneous, depending on flush completion and metadata synchronization
    # Multiple queries or a brief wait may reveal the change
    import time
    time.sleep(2) # Please wait a moment to allow the flush and metadata synchronization to complete.
    
    stats_after_insert = client.get_collection_stats(collection_name=EXERCISE_COLLECTION_NAME)
    print(f"c {stats_after_insert}")
    # The 'row_count' field typically reflects the number of entities.
    current_num_entities = int(stats_after_insert.get('row_count', 0)) # Milvus 2.x
    # 或者 desc = client.describe_collection(...); current_num_entities = desc['num_entities']

    print(f"Collection '{EXERCISE_COLLECTION_NAME}' Current number of entities: {current_num_entities}")
    if current_num_entities == NUM_ENTITIES:
        print("The number of entities matches expectations!")
    else:
        print(f"The number of entities does not match expectations (Expected: {NUM_ENTITIES}, Actual: {current_num_entities}). Flush may still be occurring in the background or other issues may exist.")

except Exception as e:
    print(f"Failed to flush collection or obtain statistics: {e}")
```

    
    Flushing Collection 'book_search'...
    The flush operation has been requested. This may take some time to complete.
    c {'row_count': 300}
    Collection 'book_search' Current number of entities: 300
    The number of entities does not match expectations (Expected: 100, Actual: 300). Flush may still be occurring in the background or other issues may exist.


 ### Hands-On: Deleting Data (by ID or filter conditions)

 Milvus supports deleting entities based on primary key IDs or scalar field filter conditions.

 - **`client.delete(collection_name, pks, filter, partition_name)`**:
     - `pks`: A list of primary key values to delete entities with these IDs.
     - `filter`: A boolean expression in string format to delete entities meeting the condition (e.g., `“publication_year < 2000”` or `"book_title like 'The Amazing%'"`).
     - At least one of `pks` or `filter` must be provided. If both are provided, entities meeting either condition will be deleted.
     - `partition_name` (optional): Restricts the delete operation to the specified partition.
 - **Logical Deletion and Compaction**:
     - Deletions in Milvus are **logical deletions (Soft Delete)**. This means data is not immediately physically removed from disk but is marked as deleted. Marked data is filtered out during searches.
     - Physical deletion and space reclamation are performed during subsequent **Compaction** operations. Compaction merges data segments, removes marked-for-deletion data, and reorganizes data to optimize storage and query performance.
     - `client.compact(collection_name)`: Manually triggers Compaction. This is an asynchronous operation.
     - `client.get_compaction_state(compaction_id)`: Query the Compaction status.
     - `client.wait_for_compaction_completed(compaction_id)`: Wait for Compaction to finish.
 - **Changes to `num_entities` after deletion**:
     - Immediately after logical deletion, querying `num_entities` may not reflect deletions.
     - After executing `flush()`, `num_entities` typically updates to reflect the number of logical deletions.
     - Only after compaction is complete will the disk space occupied by deleted data be truly reclaimed.


```python
# Before loading, you must first create indexes for vector fields (indexing details will be covered later).
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="book_embedding",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

client.create_index(
    collection_name=EXERCISE_COLLECTION_NAME,
    index_params=index_params,
    sync=False 
)
```


```python
# Data to be deleted
pks_to_delete = []
if len(inserted_pks) >= 5:
    pks_to_delete = inserted_pks[:3] # Delete the first three inserted records (by ID)
    print(f"Preparing to delete the following primary key by ID: {pks_to_delete}")
else:
    print("There are not enough primary keys to demonstrate deletion by ID.")

filter_expr_delete = "publication_year < 1990" # Remove books published before 1990
print(f"Preparing to delete based on filter criteria: '{filter_expr_delete}'")

# 0. Ensure the Collection has been loaded 
try:
    print(f"\nEnsure that Collection '{EXERCISE_COLLECTION_NAME}' is loaded...")
    # Check loading status (optional, but helpful for debugging)
    load_state_before_delete = client.get_load_state(collection_name=EXERCISE_COLLECTION_NAME)
    print(f"Collection loading status before deletion operation: {load_state_before_delete}")
    
    client.load_collection(collection_name=EXERCISE_COLLECTION_NAME)
    print(f"Collection '{EXERCISE_COLLECTION_NAME}' has been sent/confirmed.")
    # MilvusClient.load_collection() is blocking until loading completes (or times out)
    # For small or empty collections, it returns quickly
except Exception as e:
    print(f"Failed to load Collection '{EXERCISE_COLLECTION_NAME}': {e}")
    raise

# 1. Delete by ID
if pks_to_delete:
    try:
        print(f"\nDeleting data by ID...")
        del_res_ids = client.delete(
            collection_name=EXERCISE_COLLECTION_NAME,
            pks=pks_to_delete
        )
        print(f"Deletion completed via ID. Deletion count: {del_res_ids['delete_count']}") # delete_count is the number of matches found
    except Exception as e:
        print(f"Failed to delete data by ID: {e}")

# 2. Delete based on filter conditions
try:
    print(f"\nData is being deleted using the filter condition '{filter_expr_delete}'...")
    # First, check how many meet the criteria for comparison.
    query_before_delete_count = client.query(collection_name=EXERCISE_COLLECTION_NAME, 
                                             filter=filter_expr_delete, 
                                             output_fields=["book_id"])
    print(f"Before deletion, number of entities matching the condition '{filter_expr_delete}': {len(query_before_delete_count)}")

    del_res_filter = client.delete(
        collection_name=EXERCISE_COLLECTION_NAME,
        filter=filter_expr_delete
        # partition_name="fiction_books" # You can also specify partitions.
    )
    print(f"Deletion completed via filter criteria. Deletion count: {del_res_filter['delete_count']}")
except Exception as e:
    print(f"Failed to delete data based on filter conditions: {e}")

# 3. Flush Collection 以使删除生效 (更新 num_entities)
try:
    print(f"\nAfter the deletion operation, flushing Collection '{EXERCISE_COLLECTION_NAME}'...")
    client.flush(collection_name=EXERCISE_COLLECTION_NAME)
    print("Flush operation requested.")
    
    time.sleep(2) # waiting
    stats_after_delete = client.get_collection_stats(collection_name=EXERCISE_COLLECTION_NAME)
    current_num_entities_after_delete = int(stats_after_delete.get('row_count', 0))
    print(f"After flushing the Collection '{EXERCISE_COLLECTION_NAME}' current number of entities: {current_num_entities_after_delete}")
except Exception as e:
    print(f"Failed to flush Collection or obtain statistics: {e}")

# 4. (Optional Demonstration) Manually Trigger Compaction
# Compaction may take a considerable amount of time; in workshop, we may only demonstrate triggering the process without waiting for completion.
try:
    print(f"\nManually trigger Compaction for Collection'{EXERCISE_COLLECTION_NAME}'...")
    compaction_id = client.compact(collection_name=EXERCISE_COLLECTION_NAME)
    print(f"Compaction has been triggered, Compaction ID: {compaction_id}") 
    # print(f"Compaction ID: {compaction_id}") # Older versions might return ID directly

    # Check the compaction status (typically requires polling)
    # state = client.get_compaction_state(compaction_id=compaction_id.compaction_id)
    # print(f"Compaction 状态: {state}")
    
    # If you wish to wait for completion (which may take a considerable amount of time):
    # client.wait_for_compaction_completed(compaction_id=compaction_id.compaction_id)
    # print("Compaction 已完成。")
    # stats_after_compaction = client.get_collection_stats(collection_name=EXERCISE_COLLECTION_NAME)
    # print(f"Compaction 完成后 Collection 统计: {stats_after_compaction}")

except Exception as e:
    print(f"Compaction operation failed: {e}")
```

    Preparing to delete the following primary key by ID: [461486305505040590, 461486305505040591, 461486305505040592]
    Preparing to delete based on filter criteria: 'publication_year < 1990'
    
    Ensure that Collection 'book_search' is loaded...
    Collection loading status before deletion operation: {'state': <LoadState: NotLoad>}
    Collection 'book_search' has been sent/confirmed.
    
    Deleting data by ID...
    Deletion completed via ID. Deletion count: 3
    
    Data is being deleted using the filter condition 'publication_year < 1990'...
    Before deletion, number of entities matching the condition 'publication_year < 1990': 63
    Deletion completed via filter criteria. Deletion count: 63
    
    After the deletion operation, flushing Collection 'book_search'...
    Flush operation requested.
    After flushing the Collection 'book_search' current number of entities: 300
    
    Manually trigger Compaction for Collection'book_search'...
    Compaction has been triggered, Compaction ID: 461486305505496969


 ### Hands-on Exercise 2: Inserting and Deleting Data

 **Task**: 
 1.  Generate a batch of new simulated data (e.g., 50 entries) for the `book_search` Collection.
 2.  Insert this new data into the `_default` partition of `book_search`.
 3.  Record the primary keys returned after insertion.
 4.  Flush Collection。
 5.  Verify that `num_entities` has increased by the corresponding amount.
 6.  Randomly select 5 primary keys from the newly inserted data and delete the corresponding entities using these keys.
 7.  Define a filter condition (e.g., `publication_year > 2010`) and delete all entities matching this condition.
 8.  Flush the Collection again.
 9.  Verify that `num_entities` has decreased accordingly.
 10. (Optional) Delete the `fiction_books` and `non_fiction_books` partitions used for the exercise (if previously created).


```python

```


```python

```


```python

```
