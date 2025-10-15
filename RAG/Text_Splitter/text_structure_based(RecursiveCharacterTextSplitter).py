from langchain.text_splitter import RecursiveCharacterTextSplitter

text="""
    Artificial Intelligence (AI) is a transformative branch of computer science that enables machines to perform tasks traditionally requiring human intelligence, such as reasoning, learning, problem-solving, perception, and language understanding. At its core, AI combines algorithms, data, and computational power to simulate cognitive functions, allowing systems to analyze complex information, recognize patterns, and make decisions with minimal human intervention. Over the past decade, AI has evolved from basic rule-based systems to advanced machine learning and deep learning models capable of understanding natural language, interpreting images, driving autonomous vehicles, and even generating creative content. Industries worldwide, including healthcare, finance, education, manufacturing, and entertainment, have embraced AI to optimize operations, enhance customer experiences, and unlock new avenues for innovation. In healthcare, AI assists in early diagnosis, personalized treatment plans, and drug discovery, while in finance, it powers fraud detection, algorithmic trading, and risk management. AI-driven technologies also raise critical ethical, social, and legal considerations, such as data privacy, bias in algorithms, and the future of employment. Despite these challenges, AI continues to expand human potential, augment decision-making, and redefine the way we interact with technology. Its rapid advancement promises not only greater efficiency and productivity but also the potential to solve complex global problems, from climate modeling to resource management. As AI becomes increasingly integrated into everyday life, understanding its capabilities, limitations, and ethical implications is essential for harnessing its benefits responsibly and ensuring it contributes positively to society. Ultimately, AI is not just a technological tool; it represents a paradigm shift in human innovation, bridging the gap between imagination and reality, and challenging us to rethink what machines and humans can achieve together.
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    )

result=splitter.split_text(text)
print("lenght : ",len(result))
print(result)
