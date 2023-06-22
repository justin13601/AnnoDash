from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Define the database connection
engine = create_engine('sqlite:///annotations.db')
Session = sessionmaker(bind=engine)
session = Session()

# Define the database model
Base = declarative_base()


class Annotation(Base):
    __tablename__ = 'annotations'

    id = Column(Integer, primary_key=True)
    uuid = Column(String)
    itemid = Column(Integer)
    label = Column(String)
    ontology = Column(String)
    annotatedid = Column(String)
    annotatedlabel = Column(String)
    comments = Column(String)
    annotatedtime = Column(String)


# Create the table in the database (if it doesn't exist)
Base.metadata.create_all(engine)

# Load the JSON data
data = {'itemid': 220045, 'label': 'Heart Rate', 'ontology': 'loinc',
        'annotatedid': ['8867-4', '1234-5'], 'annotatedlabel': ['Heart rate', 'Some other label'],
        'comments': '', 'uuid': '6d42858a-dcdd-45f5-bb30-d39447ec44d7',
        'annotatedtime': '2023-06-02_19-20-53'}

# Save the JSON data to the database
for annotated_id, annotated_label in zip(data['annotatedid'], data['annotatedlabel']):
    annotation = Annotation(uuid=data['uuid'], itemid=data['itemid'], label=data['label'], ontology=data['ontology'],
                            annotatedid=annotated_id, annotatedlabel=annotated_label,
                            comments=data['comments'], annotatedtime=data['annotatedtime'])
    print(annotation)
    session.add(annotation)

# Commit the changes to the database
session.commit()
