# main.py

import os
from dotenv import load_dotenv
import chainlit as cl
import logging
import asyncio
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load environment FIRST ─────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

# Validate environment variables
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set in environment variables")
    raise ValueError("GEMINI_API_KEY is required")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASS]):
    logger.error("Neo4j environment variables are not set")
    raise ValueError("Neo4j credentials are required")

# Set Google API key as environment variable BEFORE importing Google modules
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# OCR (with fallback handling)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
    # Try to set tesseract path for Windows
    if os.name == 'nt':  # Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError as e:
    logger.warning(f"OCR not available: {e}")
    OCR_AVAILABLE = False

from io import BytesIO

# PDF loader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings & LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Vector store
from langchain_community.vectorstores import Neo4jVector

# Updated imports - using new LangChain syntax
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Neo4j driver
from neo4j import GraphDatabase
import uuid
from datetime import datetime

# ─── Models & Embeddings ────────────────────────────────────────────────────────
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )
    logger.info("Google AI models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Google AI models: {e}")
    raise

# ─── Neo4j Driver & Setup ──────────────────────────────────────────────────────
driver = None

def test_neo4j_connection():
    """Test Neo4j connection"""
    try:
        test_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with test_driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_result = result.single()
            test_driver.close()
            logger.info("Neo4j connection test successful")
            return True
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {e}")
        return False

def initialize_neo4j():
    """Initialize Neo4j database with constraints and indexes"""
    global driver
    try:
        # First test the connection
        if not test_neo4j_connection():
            return False
            
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # Create constraints
            try:
                session.run("CREATE CONSTRAINT heart_doc_id IF NOT EXISTS FOR (d:HeartDoc) REQUIRE d.id IS UNIQUE")
            except Exception as e:
                logger.warning(f"Constraint creation failed: {e}")
            
            try:
                session.run("CREATE CONSTRAINT patient_report_id IF NOT EXISTS FOR (r:PatientReport) REQUIRE r.id IS UNIQUE")
            except Exception as e:
                logger.warning(f"Constraint creation failed: {e}")
            
            # Create indexes for better performance
            try:
                session.run("CREATE INDEX heart_doc_text IF NOT EXISTS FOR (d:HeartDoc) ON (d.text)")
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")
                
            try:
                session.run("CREATE INDEX heart_doc_category IF NOT EXISTS FOR (d:HeartDoc) ON (d.category)")
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")
            
            logger.info("Neo4j database initialized successfully")
            return True
    except Exception as e:
        logger.error(f"Error initializing Neo4j: {e}")
        return False

def run_cypher(query: str, parameters: dict = None):
    """Execute Cypher query and return results"""
    if not driver:
        return []
    
    with driver.session() as session:
        try:
            result = session.run(query, parameters or {})
            return list(result)
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []

def store_patient_report(report_text: str, summary: str, report_type: str = "cardiac"):
    """Store patient report in Neo4j"""
    if not driver:
        return None
        
    report_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    query = """
    CREATE (r:PatientReport {
        id: $report_id,
        text: $report_text,
        summary: $summary,
        type: $report_type,
        timestamp: $timestamp,
        analyzed: true
    })
    RETURN r.id as id
    """
    
    parameters = {
        "report_id": report_id,
        "report_text": report_text[:5000],  # Limit text length
        "summary": summary[:2000],  # Limit summary length
        "report_type": report_type,
        "timestamp": timestamp
    }
    
    result = run_cypher(query, parameters)
    return report_id if result else None

# ─── PDF Ingestion & Knowledge Base Setup ──────────────────────────────────────
PDF_FOLDER = "./pdf"

def categorize_content(text: str) -> str:
    """Categorize content based on keywords"""
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in ['diet', 'nutrition', 'food', 'meal', 'eating']):
        return 'diet'
    elif any(keyword in text_lower for keyword in ['exercise', 'activity', 'physical', 'workout']):
        return 'exercise'
    elif any(keyword in text_lower for keyword in ['medication', 'drug', 'prescription', 'medicine']):
        return 'medication'
    elif any(keyword in text_lower for keyword in ['precaution', 'warning', 'avoid', 'risk']):
        return 'precautions'
    elif any(keyword in text_lower for keyword in ['symptom', 'sign', 'diagnosis', 'condition']):
        return 'symptoms'
    else:
        return 'general'

def load_and_split_pdfs():
    """Load PDFs and split into chunks with enhanced text splitting"""
    docs = []
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        logger.info(f"Created PDF folder: {PDF_FOLDER}")
        return []
    
    # Check if PDF folder has files
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in {PDF_FOLDER}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    # Enhanced text splitter with better parameters for medical content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced chunk size for faster processing
        chunk_overlap=100,  # Reduced overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    for fname in pdf_files:
        try:
            pdf_path = os.path.join(PDF_FOLDER, fname)
            logger.info(f"Loading PDF: {fname}")
            
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            
            # Add metadata to each document
            for doc in pdf_docs:
                doc.metadata['source_file'] = fname
                doc.metadata['content_type'] = 'medical_guideline'
            
            docs.extend(pdf_docs)
            logger.info(f"Successfully loaded PDF: {fname} ({len(pdf_docs)} pages)")
            
        except Exception as e:
            logger.error(f"Error loading {fname}: {e}")
            continue
    
    if not docs:
        logger.warning("No PDF documents loaded successfully")
        return []
    
    # Split documents
    logger.info(f"Splitting {len(docs)} documents into chunks...")
    split_docs = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(split_docs)} chunks")
    
    return split_docs

def setup_vector_store_and_graph(docs):
    """Setup both vector store and graph database with categorized content"""
    if not docs or not driver:
        logger.error("No documents or driver not available")
        return None
    
    try:
        logger.info("Setting up vector store...")
        
        # Store in Neo4j Vector with smaller batch size
        vector_store = Neo4jVector.from_documents(
            documents=docs[:100],  # Limit initial load for faster startup
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASS,
            index_name="heart_docs_vector",
            node_label="HeartDoc",
            text_node_property="text",
            embedding_node_property="embedding"
        )
        
        logger.info("Vector store created, adding categorized content...")
        
        # Additionally store categorized content in graph (process in batches)
        batch_size = 50
        for i in range(0, min(len(docs), 100), batch_size):
            batch_docs = docs[i:i+batch_size]
            
            with driver.session() as session:
                for j, doc in enumerate(batch_docs):
                    category = categorize_content(doc.page_content)
                    
                    # Create nodes with categories and relationships
                    query = """
                    MERGE (d:HeartDoc {id: $doc_id})
                    SET d.text = $text,
                        d.category = $category,
                        d.source = $source,
                        d.chunk_index = $chunk_index,
                        d.created_at = datetime()
                    
                    // Create category nodes and relationships
                    MERGE (c:Category {name: $category})
                    MERGE (d)-[:BELONGS_TO]->(c)
                    
                    RETURN d.id as doc_id
                    """
                    
                    parameters = {
                        "doc_id": f"doc_{i+j}_{uuid.uuid4().hex[:8]}",
                        "text": doc.page_content[:2000],  # Limit text length
                        "category": category,
                        "source": doc.metadata.get('source_file', 'unknown'),
                        "chunk_index": i+j
                    }
                    
                    session.run(query, parameters)
            
            logger.info(f"Processed batch {i//batch_size + 1}")
        
        logger.info("Vector store and graph database setup completed")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        traceback.print_exc()
        return None

# ─── Enhanced Prompt Templates (Updated for new LangChain) ─────────────────────

# Medical Report Analysis Prompt
analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert cardiology assistant. Analyze the following clinical report comprehensively.

Clinical Report:
{report_text}

Please provide a detailed analysis including:

**CLINICAL FINDINGS:**
- Primary cardiac conditions identified
- Key vital signs and measurements
- Laboratory values and their significance
- Risk factors present

**DIAGNOSTIC SUMMARY:**
- Main diagnosis or suspected conditions
- Severity assessment
- Areas of concern

**RECOMMENDATIONS:**
- Immediate actions needed
- Follow-up requirements
- Monitoring parameters

Format your response clearly with the sections above. Be precise and medical in your language.
""")

# Diet and Lifestyle Recommendations Prompt
diet_prompt = ChatPromptTemplate.from_template("""
Based on the patient's clinical analysis and evidence-based guidelines, provide personalized recommendations.

**PATIENT ANALYSIS:**
{summary}

**EVIDENCE-BASED GUIDELINES:**
{retrieved_texts}

**PATIENT'S CURRENT CONTEXT:**
{chat_history}

Please provide:

**PERSONALIZED DIET PLAN:**
1. Daily meal structure with specific recommendations
2. Portion sizes and timing
3. Key nutrients to focus on
4. Sample meal ideas

**LIFESTYLE MODIFICATIONS:**
1. Physical activity recommendations
2. Stress management techniques
3. Sleep and recovery guidelines
4. Monitoring recommendations

**PRECAUTIONS & RESTRICTIONS:**
1. Foods to avoid or limit
2. Activities to restrict
3. Warning signs to watch for
4. When to seek immediate medical attention

**FOLLOW-UP PLAN:**
1. Timeline for reassessment
2. Key parameters to monitor
3. Recommended specialist consultations

Make recommendations specific, actionable, and evidence-based. Consider the patient's current condition severity.
""")

# Chat Conversation Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized cardiac health assistant. You have access to:
    1. The patient's analyzed report and summary
    2. Evidence-based medical guidelines from your knowledge base
    3. Previous conversation context

    Always:
    - Provide evidence-based advice
    - Reference relevant guidelines when applicable
    - Be supportive but emphasize professional medical consultation
    - Ask clarifying questions when needed
    - Maintain conversation context
    
    Patient Summary: {patient_summary}
    Context: {context}
    Chat History: {chat_history}
    """),
    ("human", "{question}")
])

# Create chains using new syntax (no more LLMChain)
analysis_chain = analysis_prompt | llm | StrOutputParser()
diet_chain = diet_prompt | llm | StrOutputParser()
chat_chain = chat_prompt | llm | StrOutputParser()

# ─── Enhanced Retrieval Functions ──────────────────────────────────────────────

def retrieve_relevant_guidelines(summary: str, categories: list = None, k: int = 3):
    """Retrieve relevant guidelines using both semantic and category-based search"""
    if not driver:
        return []
        
    if not categories:
        categories = ['diet', 'precautions', 'exercise', 'medication']
    
    # Semantic search query
    category_filter = " OR ".join([f"d.category = '{cat}'" for cat in categories])
    
    query = f"""
    MATCH (d:HeartDoc)
    WHERE ({category_filter})
    AND (
        toLower(d.text) CONTAINS toLower($keyword1) OR
        toLower(d.text) CONTAINS toLower($keyword2) OR
        toLower(d.text) CONTAINS toLower($keyword3) OR
        toLower(d.text) CONTAINS toLower($keyword4)
    )
    RETURN d.text AS text, d.category AS category, d.source AS source
    ORDER BY d.category
    LIMIT $limit
    """
    
    # Extract keywords from summary for better matching
    keywords = extract_medical_keywords(summary)
    
    parameters = {
        "keyword1": keywords.get('condition', 'cardiac'),
        "keyword2": keywords.get('symptom', 'heart'),
        "keyword3": keywords.get('risk', 'cholesterol'),
        "keyword4": keywords.get('treatment', 'lifestyle'),
        "limit": k
    }
    
    results = run_cypher(query, parameters)
    return results

def extract_medical_keywords(text: str) -> dict:
    """Extract medical keywords from text for better retrieval"""
    text_lower = text.lower()
    keywords = {}
    
    # Common cardiac conditions
    conditions = ['hypertension', 'diabetes', 'cholesterol', 'arrhythmia', 'coronary', 'cardiac']
    for condition in conditions:
        if condition in text_lower:
            keywords['condition'] = condition
            break
    
    # Common symptoms
    symptoms = ['chest pain', 'shortness of breath', 'fatigue', 'palpitations']
    for symptom in symptoms:
        if symptom in text_lower:
            keywords['symptom'] = symptom
            break
    
    # Risk factors
    risks = ['smoking', 'obesity', 'stress', 'sedentary']
    for risk in risks:
        if risk in text_lower:
            keywords['risk'] = risk
            break
    
    return keywords

# ─── OCR Processing Function ───────────────────────────────────────────────────

async def process_image_ocr(image_bytes: bytes) -> str:
    """Process image with OCR in a thread-safe manner"""
    if not OCR_AVAILABLE:
        raise Exception("OCR is not available. Please install pytesseract and Tesseract OCR.")
    
    try:
        # Process in a separate thread to avoid blocking
        def _ocr_process():
            img = Image.open(BytesIO(image_bytes))
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return pytesseract.image_to_string(img)
        
        result = await asyncio.to_thread(_ocr_process)
        return result.strip()
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise Exception(f"OCR failed: {str(e)}")

# ─── File Processing Functions ─────────────────────────────────────────────────


async def process_uploaded_file(file) -> tuple[str, bool]:
    """Process uploaded file and return (text, is_report)"""
    try:
        file_name = getattr(file, 'name', '')
        file_type = getattr(file, 'type', '')
        
        logger.info(f"Processing file: {file_name} of type: {file_type}")
        
        # Get file content first
        if hasattr(file, 'content'):
            file_content = file.content
        elif hasattr(file, 'read'):
            file_content = await file.read()
        else:
            raise Exception("Unable to read file content")
        
        # Ensure we have bytes
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')
        
        # Handle image files (OCR) - check both MIME type and file extension
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        image_mime_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff', 'image/gif']
        
        is_image = (
            (file_type and any(mime in file_type.lower() for mime in image_mime_types)) or
            (file_name and any(file_name.lower().endswith(ext) for ext in image_extensions))
        )
        
        if is_image:
            if not OCR_AVAILABLE:
                raise Exception("OCR not available. Please install pytesseract and Tesseract OCR.")
            
            text = await process_image_ocr(file_content)
            
            if not text or len(text.strip()) < 10:
                raise Exception("No readable text could be extracted from the image. Please try a clearer image or check if the image contains text.")
            
            return text, True
        
        # Handle PDF files
        pdf_extensions = ['.pdf']
        pdf_mime_types = ['application/pdf']
        
        is_pdf = (
            (file_type and any(mime in file_type.lower() for mime in pdf_mime_types)) or
            (file_name and any(file_name.lower().endswith(ext) for ext in pdf_extensions))
        )
        
        if is_pdf:
            # Create temp directory if it doesn't exist
            temp_dir = "./temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_path = os.path.join(temp_dir, f"temp_{uuid.uuid4().hex[:8]}.pdf")
            
            try:
                # Write file content
                with open(temp_path, 'wb') as f:
                    f.write(file_content)
                
                # Process PDF
                def _load_pdf():
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    return "\n".join([doc.page_content for doc in docs])
                
                text = await asyncio.to_thread(_load_pdf)
                
                if not text or len(text.strip()) < 10:
                    raise Exception("No readable text could be extracted from the PDF.")
                
                return text, True
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Handle text files
        text_extensions = ['.txt', '.csv']
        text_mime_types = ['text/plain', 'text/csv', 'application/csv']
        
        is_text = (
            (file_type and any(mime in file_type.lower() for mime in text_mime_types)) or
            (file_name and any(file_name.lower().endswith(ext) for ext in text_extensions))
        )
        
        if is_text:
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = file_content.decode('latin-1')
                except UnicodeDecodeError:
                    text = file_content.decode('utf-8', errors='ignore')
            
            if not text or len(text.strip()) < 10:
                raise Exception("The text file appears to be empty or contains no readable content.")
            
            return text, True
        
        # If we reach here, unsupported file type
        supported_types = "PNG, JPG, JPEG, PDF, TXT"
        raise Exception(
            f"Unsupported file type. File: '{file_name}' (Type: '{file_type}')\n"
            f"Supported formats: {supported_types}"
        )
            
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise


async def process_image_ocr(image_bytes: bytes) -> str:
    """Process image with OCR in a thread-safe manner"""
    if not OCR_AVAILABLE:
        raise Exception("OCR is not available. Please install pytesseract and Tesseract OCR.")
    
    try:
        # Process in a separate thread to avoid blocking
        def _ocr_process():
            try:
                img = Image.open(BytesIO(image_bytes))
                
                # Convert to RGB if needed for better OCR
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # OCR configuration for better text extraction
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}"\'-+=/\n '
                
                # Try different OCR configurations
                try:
                    # First try with default config
                    result = pytesseract.image_to_string(img, config='--psm 6')
                    if len(result.strip()) > 20:  # If we got decent results
                        return result
                    
                    # Try with different page segmentation mode
                    result = pytesseract.image_to_string(img, config='--psm 3')
                    if len(result.strip()) > 10:
                        return result
                    
                    # Last attempt with custom config
                    result = pytesseract.image_to_string(img, config=custom_config)
                    return result
                    
                except Exception as ocr_error:
                    logger.warning(f"OCR processing failed with config: {ocr_error}")
                    # Fallback to basic OCR
                    return pytesseract.image_to_string(img)
                
            except Exception as img_error:
                logger.error(f"Image processing error: {img_error}")
                raise Exception(f"Failed to process image: {str(img_error)}")
        
        result = await asyncio.to_thread(_ocr_process)
        return result.strip()
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise Exception(f"OCR failed: {str(e)}")


# ─── Cleanup function ──────────────────────────────────────────────────────────

def cleanup_resources():
    """Clean up resources on shutdown"""
    global driver
    if driver:
        driver.close()
        logger.info("Neo4j driver closed")

# ─── Chainlit Events ───────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    """Initialize the application"""
    try:
        # Show initial loading message
        loading_msg = cl.Message("🔄 **Initializing Cardiac Health Assistant...**")
        await loading_msg.send()
        
        # Initialize Neo4j
        loading_msg.content = "🔄 **Connecting to database...**"
        await loading_msg.update()
        neo4j_success = initialize_neo4j()
        
        if not neo4j_success:
            loading_msg.content = "❌ **Database connection failed. Running in limited mode.**"
            await loading_msg.update()
            cl.user_session.set("vector_store", None)
            cl.user_session.set("chat_history", [])
            cl.user_session.set("patient_summary", None)
            cl.user_session.set("report_id", None)
            return
        
        # Load and process PDFs
        loading_msg.content = "🔄 **Loading medical guidelines...**"
        await loading_msg.update()
        docs = load_and_split_pdfs()
        
        if docs and neo4j_success:
            loading_msg.content = "🔄 **Setting up knowledge base...**"
            await loading_msg.update()
            store = setup_vector_store_and_graph(docs)
            
            if store:
                cl.user_session.set("vector_store", store)
                cl.user_session.set("chat_history", [])
                cl.user_session.set("patient_summary", None)
                cl.user_session.set("report_id", None)
                
                ocr_status = "✅ OCR enabled" if OCR_AVAILABLE else "⚠️ OCR not available"
                
                loading_msg.content = (
                    f"🫀 **Cardiac Health Assistant Ready!**\n\n"
                    f"✅ Loaded {len(docs)} medical guideline chunks\n"
                    f"✅ Vector search enabled\n"
                    f"✅ Graph database ready\n"
                    f"{ocr_status}\n\n"
                    f"**You can:**\n"
                    f"• Upload a PDF report for analysis\n"
                    f"• Upload an image of your report {'(OCR enabled)' if OCR_AVAILABLE else '(install pytesseract for OCR)'}\n"
                    f"• Paste your report text directly\n"
                    f"• Ask questions about cardiac health\n\n"
                    f"*Please upload your cardiac report to get personalized recommendations.*"
                )
                await loading_msg.update()
            else:
                cl.user_session.set("vector_store", None)
                cl.user_session.set("chat_history", [])
                cl.user_session.set("patient_summary", None)
                cl.user_session.set("report_id", None)
                loading_msg.content = (
                    "⚠️ **Partial Setup Complete**\n\n"
                    "Vector store setup failed, but basic analysis is available."
                )
                await loading_msg.update()
        else:
            cl.user_session.set("vector_store", None)
            cl.user_session.set("chat_history", [])
            cl.user_session.set("patient_summary", None)
            cl.user_session.set("report_id", None)
            loading_msg.content = (
                "🫀 **Basic Mode Active**\n\n"
                "Database connection failed or no PDF guidelines found.\n"
                "You can still upload reports for basic analysis."
            )
            await loading_msg.update()
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        traceback.print_exc()
        cl.user_session.set("vector_store", None)
        cl.user_session.set("chat_history", [])
        cl.user_session.set("patient_summary", None)
        cl.user_session.set("report_id", None)
        await cl.Message(
            f"⚠️ **Setup Error**: {str(e)}\n\n"
            "Please check your environment variables and try again."
        ).send()

@cl.on_message
async def handle_message(msg):
    """Handle all types of messages"""
    try:
        vector_store = cl.user_session.get("vector_store")
        chat_history = cl.user_session.get("chat_history", [])
        patient_summary = cl.user_session.get("patient_summary")
        
        report_text = msg.content if msg.content else ""
        is_report_upload = False
        
        # ─── Handle File Uploads ───────────────────────────────────────────────────
        if hasattr(msg, "elements") and msg.elements and len(msg.elements) > 0:
            try:
                file = msg.elements[0]  # Get the first file
                file_name = getattr(file, 'name', 'unknown')
                file_type = getattr(file, 'type', 'unknown')
                
                # Show processing message with file info
                processing_msg = cl.Message(f"🔄 **Processing your file:** {file_name} ({file_type})")
                await processing_msg.send()
                
                # Process the file
                report_text, is_report_upload = await process_uploaded_file(file)
                
                # Update processing message
                processing_msg.content = f"✅ **File processed successfully:** {file_name}"
                await processing_msg.update()
                
                if report_text and len(report_text.strip()) > 50:
                    preview_text = report_text.strip()
                    if len(preview_text) > 500:
                        preview_text = preview_text[:500] + "..."
                    await cl.Message(f"📄 **Extracted Text Preview:**\n```\n{preview_text}\n```").send()
                else:
                    await cl.Message(
                        f"⚠️ **Warning:** Only extracted {len(report_text.strip())} characters from the file.\n"
                        "Please ensure:\n"
                        "• The image contains clear, readable text\n"
                        "• The PDF is not password protected\n"
                        "• The file is not corrupted"
                    ).send()
                    if len(report_text.strip()) < 20:  # Too little text to be useful
                        return
                    
            except Exception as e:
                logger.error(f"File upload error: {e}")
                error_msg = str(e)
                if "OCR not available" in error_msg:
                    await cl.Message(
                        "❌ **OCR Not Available**\n\n"
                        "To process image files, please install:\n"
                        "```bash\n"
                        "pip install pytesseract\n"
                        "```\n"
                        "And download Tesseract OCR from: https://github.com/tesseract-ocr/tesseract\n\n"
                        "Alternatively, you can:\n"
                        "• Convert your image to text and paste it\n"
                        "• Upload a PDF version instead"
                    ).send()
                else:
                    await cl.Message(f"❌ **File Processing Error**: {error_msg}").send()
                return
        
        # Rest of your existing code for processing reports and chat...
        # [Continue with the existing logic from your original handle_message function]
        
    except Exception as e:
        logger.error(f"Unexpected error in handle_message: {e}")
        logger.error(traceback.format_exc())
        await cl.Message(f"❌ **Unexpected Error**: {str(e)}").send()
    """Handle all types of messages"""
    try:
        vector_store = cl.user_session.get("vector_store")
        chat_history = cl.user_session.get("chat_history", [])
        patient_summary = cl.user_session.get("patient_summary")
        
        report_text = msg.content if msg.content else ""
        is_report_upload = False
        
        # ─── Handle File Uploads ───────────────────────────────────────────────────
        if hasattr(msg, "elements") and msg.elements and len(msg.elements) > 0:
            try:
                file = msg.elements[0]  # Get the first file
                
                # Show processing message
                processing_msg = cl.Message("🔄 **Processing your file...**")
                await processing_msg.send()
                
                # Process the file
                report_text, is_report_upload = await process_uploaded_file(file)
                
                # Update processing message
                processing_msg.content = "✅ **File processed successfully!**"
                await processing_msg.update()
                
                if report_text and len(report_text.strip()) > 50:
                    preview_text = report_text.strip()
                    if len(preview_text) > 500:
                        preview_text = preview_text[:500] + "..."
                    await cl.Message(f"📄 **Extracted Text Preview:**\n```\n{preview_text}\n```").send()
                else:
                    await cl.Message("⚠️ Very little text extracted. Please ensure the file contains readable text.").send()
                    return
                    
            except Exception as e:
                logger.error(f"File upload error: {e}")
                await cl.Message(f"❌ **File Processing Error**: {str(e)}").send()
                return
        
        # ─── Process Report Upload ─────────────────────────────────────────────────
        if is_report_upload or (not patient_summary and len(report_text.strip()) > 100):
            if not report_text.strip():
                await cl.Message("❌ No text found. Please provide a valid report.").send()
                return
            
            try:
                analysis_msg = cl.Message("🔬 **Analyzing your cardiac report...**")
                await analysis_msg.send()
                
                # Analyze the report using new chain syntax
                summary = await asyncio.to_thread(
                    analysis_chain.invoke,
                    {"report_text": report_text}
                )
                
                # Store in Neo4j
                report_id = store_patient_report(report_text, summary, "cardiac")
                
                # Store in session
                cl.user_session.set("patient_summary", summary)
                cl.user_session.set("report_id", report_id)
                
                analysis_msg.content = "✅ **Analysis complete!**"
                await analysis_msg.update()
                await cl.Message(f"📋 **Clinical Analysis:**\n\n{summary}").send()
                
                # Get personalized recommendations
                if vector_store:
                    guidelines_msg = cl.Message("🔍 **Retrieving personalized guidelines...**")
                    await guidelines_msg.send()
                    
                    guidelines = retrieve_relevant_guidelines(summary)
                    retrieved_texts = "\n\n".join([f"**{g['category'].title()}**: {g['text']}" for g in guidelines])
                    
                    if retrieved_texts:
                        diet_output = await asyncio.to_thread(
                            diet_chain.invoke,
                            {
                                "summary": summary,
                                "retrieved_texts": retrieved_texts,
                                "chat_history": "\n".join(chat_history[-3:])
                            }
                        )
                        
                        guidelines_msg.content = "✅ **Guidelines retrieved!**"
                        await guidelines_msg.update()
                        await cl.Message(f"🍎 **Personalized Recommendations:**\n\n{diet_output}").send()
                    else:
                        guidelines_msg.content = "⚠️ No specific guidelines found. Using general recommendations."
                        await guidelines_msg.update()
                        # Provide general recommendations even without specific guidelines
                        general_recommendations = await asyncio.to_thread(
                            diet_chain.invoke,
                            {
                                "summary": summary,
                                "retrieved_texts": "General cardiac health guidelines apply.",
                                "chat_history": "\n".join(chat_history[-3:])
                            }
                        )
                        await cl.Message(f"🍎 **General Recommendations:**\n\n{general_recommendations}").send()
                
                await cl.Message(
                    "✅ **Analysis Complete!**\n\n"
                    "You can now ask me specific questions about:\n"
                    "• Diet modifications\n"
                    "• Exercise recommendations\n"
                    "• Lifestyle changes\n"
                    "• Medication queries\n"
                    "• Symptom monitoring\n\n"
                    "*How can I help you further with your cardiac health?*"
                ).send()
                
            except Exception as e:
                logger.error(f"Report analysis error: {e}")
                logger.error(traceback.format_exc())
                await cl.Message(f"❌ **Analysis Error**: {str(e)}").send()
                return
        
        # ─── Handle Chat Questions ─────────────────────────────────────────────────
        else:
            if not patient_summary:
                await cl.Message(
                    "👋 **Hello!** I'm your cardiac health assistant.\n\n"
                    "To provide personalized advice, please first upload your cardiac report (PDF or image) "
                    "or paste your report text.\n\n"
                    "After analysis, I can answer specific questions about diet, exercise, medications, and lifestyle modifications."
                ).send()
                return
            
            try:
                # Add to chat history
                chat_history.append(f"Patient: {msg.content}")
                
                # Get relevant context if available
                context = ""
                if vector_store:
                    guidelines = retrieve_relevant_guidelines(f"{patient_summary} {msg.content}")
                    if guidelines:
                        context = "\n".join([f"{g['category']}: {g['text'][:200]}..." for g in guidelines[:3]])
                
                # Generate response using new chain syntax
                response = await asyncio.to_thread(
                    chat_chain.invoke,
                    {
                        "question": msg.content,
                        "patient_summary": patient_summary or "",
                        "context": context,
                        "chat_history": "\n".join(chat_history[-6:])
                    }
                )
                
                chat_history.append(f"Assistant: {response}")
                cl.user_session.set("chat_history", chat_history)
                
                await cl.Message(response).send()
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                logger.error(traceback.format_exc())
                await cl.Message(f"❌ **Error**: {str(e)}").send()
                
    except Exception as e:
        logger.error(f"Unexpected error in handle_message: {e}")
        logger.error(traceback.format_exc())
        await cl.Message(f"❌ **Unexpected Error**: {str(e)}").send()

@cl.on_chat_end
async def end():
    """Clean up resources when chat ends"""
    try:
        cleanup_resources()
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# ─── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs("./temp", exist_ok=True)
    
    # Register cleanup handler
    import atexit
    atexit.register(cleanup_resources)
    
    logger.info("Starting Cardiac Health Assistant...")
    
    # Run the chainlit app
    # Note: This will be handled by chainlit run command
    pass