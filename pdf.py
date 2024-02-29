import os
from llama_index import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers import PDFReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index

pdf_path = os.path.join("data", "Differential_Equations.pdf")
Differential_Equations_pdf = PDFReader().load_data(file=pdf_path)
Differential_Equations_index = get_index(Differential_Equations_pdf, "Differential_Equations")
Differential_Equations_engine = Differential_Equations_index.as_query_engine()