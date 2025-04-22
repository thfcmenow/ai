import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';
import { env } from 'process';
import dotenv from 'dotenv';
dotenv.config();



// Add this in your embeddings.js file where you initialize ONNX
import * as ort from 'onnxruntime-node';



// Use import.meta.url to get the current module's URL
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class EmbeddingManager {
  constructor() {
    this.articles = [];
    this.embeddings = [];
    this.pipeline = null;
    this.textGenerationPipeline = null;
    this.embeddingsFile = path.join(__dirname, 'embeddings.json');
    this.embeddingsVersion = 1; // For future compatibility
    this.openai = new OpenAI({
       apiKey: process.env.AIKEY // Correct way to access the environment variable
     //apiKey: process.env.GROK,
    // baseURL: 'https://api.x.ai/v1', // xAI's API endpoint
    });
  }

  // Format log messages with timestamps
  logWithTime(message, type = 'info') {
    return false
    const timestamp = new Date().toISOString();
    const prefix = type === 'error' ? '❌ ERROR:' : '✓ INFO:';
    // console[type === 'error' ? 'error' : 'log'](`[${timestamp}] ${prefix} ${message}`);
  }

  // Load embeddings from disk
  async loadEmbeddings() {
    try {
      const data = await fs.readFile(this.embeddingsFile, 'utf-8');
      const loadedData = JSON.parse(data);
      
      // Check version compatibility
      if (!loadedData.version || loadedData.version !== this.embeddingsVersion) {
        this.logWithTime('Embeddings version mismatch or missing. Regenerating...', 'info');
        return false;
      }
      
      // Store the embeddings
      this.embeddings = loadedData.embeddings;
      
      // Validate the loaded embeddings have all necessary properties
      const isValid = this.embeddings.every(item => 
        item.embedding && 
        item.content && 
        item.did && 
        Array.isArray(item.cats) &&
        item.article
      );
      
      if (!isValid) {
        this.logWithTime('Loaded embeddings are missing required properties. Regenerating...', 'info');
        this.embeddings = [];
        return false;
      }
      
      this.logWithTime(`Embeddings loaded from disk successfully (${this.embeddings.length} items)`);
      return true;
    } catch (error) {
      this.logWithTime('No existing embeddings found on disk or failed to load. Will generate new embeddings...', 'info');
      return false;
    }
  }

  // Save embeddings to disk
  async saveEmbeddings() {
    try {
      // Include version and timestamp in the saved file
      const dataToSave = {
        version: this.embeddingsVersion,
        timestamp: new Date().toISOString(),
        count: this.embeddings.length,
        embeddings: this.embeddings
      };
      
      await fs.writeFile(this.embeddingsFile, JSON.stringify(dataToSave, null, 2), 'utf-8');
      this.logWithTime(`Embeddings saved to disk successfully (${this.embeddings.length} items)`);
      return true;
    } catch (error) {
      this.logWithTime(`Error saving embeddings: ${error.message}`, 'error');
      return false;
    }
  }

  // Initialize embedding model and text generation model
  async init() {
    try {
      // Dynamic import for the transformers package
      this.logWithTime('Loading transformers package...');
      const { pipeline: pipelineFunc, env: transformersEnv } = await import('@xenova/transformers');
      
      // Configure transformers to be less verbose
      transformersEnv.quiet = true;
      transformersEnv.silent = true;
      transformersEnv.logger = {
        info: () => {}, // Suppress info logs
        warn: (msg) => {
          // Only log certain warnings
          if (!msg.includes('Removing initializer')) {
            console.warn(msg);
          }
        },
        error: () => {}, // Suppress error logs
        debug: () => {}, // Suppress debug logs
        // error: (msg) => console.error(msg)
      };
      
      // Initialize embedding model
      this.logWithTime('Initializing embedding model...');
      this.pipeline = await pipelineFunc('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        cache_dir: '/tmp/xenova-cache', // Use Vercel's writable /tmp directory
        quantized: true, // Use quantized model for smaller memory footprint
        progress_callback: null // Disable progress logs
      });
      
      // Initialize text generation model (lightweight)
      // this.logWithTime('Initializing text generation model...');
      /* = await pipelineFunc('text-generation', 'Xenova/gpt2', {
        cache_dir: '/tmp/xenova-cache', // Use Vercel's writable /tmp directory
        quantized: true, // Use quantized model for smaller memory footprint
        progress_callback: null // Disable progress logs
      }); */
      
      // Load articles and generate/load embeddings
      await this.loadArticles();
      
      // Try to load embeddings first, generate only if needed
      const embeddingsLoaded = await this.loadEmbeddings();
      
      // Check if we need to regenerate embeddings (if none loaded or article count mismatch)
      const articleIds = new Set(this.articles.map(article => article.did));
      const embeddingIds = new Set(this.embeddings.map(item => item.did));
      
      // Compare article IDs with embedding IDs
      const needsRegeneration = !embeddingsLoaded || 
                               articleIds.size !== embeddingIds.size ||
                               ![...articleIds].every(id => embeddingIds.has(id));
      
      if (needsRegeneration) {
        this.logWithTime('Regenerating embeddings due to article changes or missing embeddings');
        this.embeddings = []; // Clear any partial embeddings
        await this.generateEmbeddings();
      }
      
      this.logWithTime('Initialization complete');
    } catch (error) {
      this.logWithTime(`Error during initialization: ${error.message}`, 'error');
      throw error;
    }
  }

  // Load JSON files from articles folder
  async loadArticles() {
    try {
      const articleDir = path.join(__dirname, 'articles');
      const files = await fs.readdir(articleDir);
      let loadedCount = 0;
      let skippedCount = 0;
      
      // Clear existing articles
      this.articles = [];
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          try {
            const content = await fs.readFile(path.join(articleDir, file), 'utf-8');
            const article = JSON.parse(content);
            
            // Validate article content
            if (!article.title || typeof article.title !== 'string' || article.title.trim() === '') {
              this.logWithTime(`Skipping article ${file}: Missing or empty title`, 'error');
              skippedCount++;
              continue;
            }
            
            if (!article.body || typeof article.body !== 'string' || article.body.trim() === '') {
              this.logWithTime(`Skipping article ${file}: Missing or empty body`, 'error');
              skippedCount++;
              continue;
            }
            
            // Verify did exists and is unique
            if (!article.did) {
              this.logWithTime(`Skipping article ${file}: Missing document ID (did)`, 'error');
              skippedCount++;
              continue;
            }
            
            // Trim and clean content
            const cleanTitle = article.title.trim();
            const cleanContent = this.cleanHtmlContent(article.body.trim());
            
            // Add article with validated content
            this.articles.push({
              title: cleanTitle,
              body: cleanContent,
              did: article.did,
              sitename: article.site || 'Unknown Site',
              cats: Array.isArray(article.cats) ? article.cats : []
            });
            loadedCount++;
          } catch (fileError) {
            this.logWithTime(`Failed to load article ${file}: ${fileError.message}`, 'error');
            skippedCount++;
          }
        }
      }
      
      this.logWithTime(`Loaded ${loadedCount} articles successfully, skipped ${skippedCount} invalid articles`);
      
      if (this.articles.length === 0) {
        this.logWithTime('Warning: No valid articles were loaded. Search functionality will not work.', 'error');
      }
    } catch (error) {
      this.logWithTime(`Error loading articles: ${error.message}`, 'error');
      throw error;
    }
  }

  // Clean HTML content
  cleanHtmlContent(html) {
    if (!html) return '';
    return html
      .replace(/<[^>]*>/g, ' ')             // Remove HTML tags
      .replace(/&[^;]+;/g, match => {       // Handle common HTML entities
        const entities = {
          '&ldquo;': '"',
          '&rdquo;': '"',
          '&nbsp;': ' ',
          '&#39;': "'",
          '&quot;': '"',
          '&amp;': '&',
          '&lt;': '<',
          '&gt;': '>'
        };
        return entities[match] || ' ';
      })
      .replace(/\s+/g, ' ')                 // Normalize whitespace
      .trim();
  }

  // Generate embeddings for all articles
  async generateEmbeddings() {
    try {
      if (!this.pipeline) {
        this.logWithTime('Embedding pipeline not initialized', 'error');
        throw new Error('Embedding pipeline must be initialized before generating embeddings');
      }
      
      this.logWithTime(`Generating embeddings for ${this.articles.length} articles...`);
      let successCount = 0;
      this.embeddings = []; // Clear existing embeddings
      
      for (const article of this.articles) {
        try {
          // Make sure we have content to embed
          if (!article.title && !article.body) {
            this.logWithTime(`Skipping article with no content`, 'error');
            continue;
          }
          
          const text = `${article.title || ''} ${article.body || ''}`;
          const embedding = await this.pipeline(text, { 
            pooling: 'mean', 
            normalize: true,
            progress_callback: null // Disable progress logs
          });
          
          this.embeddings.push({
            embedding: embedding.data,
            content: article.body,
            did: article.did,
            cats: article.cats,
            article: {
              title: article.title,
              body: article.body,
              did: article.did,
              sitename: article.sitename,
              cats: article.cats
            }
          });
          successCount++;
          
          // Log progress for large sets of articles
          if (successCount % 10 === 0) {
            this.logWithTime(`Generated ${successCount}/${this.articles.length} embeddings...`);
          }
        } catch (embeddingError) {
          this.logWithTime(`Failed to generate embedding for article "${article.title}": ${embeddingError.message}`, 'error');
        }
      }
      
      this.logWithTime(`Generated embeddings for ${successCount}/${this.articles.length} articles`);
      
      if (this.embeddings.length === 0) {
        this.logWithTime('Warning: No embeddings were generated. Search functionality will not work.', 'error');
      } else {
        // Save embeddings to disk after generation
        await this.saveEmbeddings();
      }
    } catch (error) {
      this.logWithTime(`Error generating embeddings: ${error.message}`, 'error');
      throw error;
    }
  }

  // Cosine similarity for comparing embeddings
  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    return dotProduct / (normA * normB);
  }

  // Strip HTML for generating answers
  stripHtml(html) {
    if (!html) return '';
    return html
      .replace(/<[^>]*>/g, ' ')
      .replace(/&[a-z0-9#]+;/gi, match => {
        const entities = {
          '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
          '&quot;': '"', '&#39;': "'", '&ldquo;': '"', '&rdquo;': '"'
        };
        return entities[match] || ' ';
      })
      .replace(/\s+/g, ' ')
      .trim();
  }

  // Generate an answer based on relevant articles
  async generateAnswer(relevantArticles, query) {
    try {
      if (!relevantArticles || relevantArticles.length === 0) {
        this.logWithTime('No relevant articles found for generating answer', 'error');
        return 'No relevant information found to answer your query.';
      }
      
      // Extract cleaned content from articles
      const context = relevantArticles
        .map((article, index) => {
          // Clean content if needed
          const cleanContent = article.content || article.body || '';
          const content = this.stripHtml(cleanContent).substring(0, 7000);
          return `ARTICLE ${index + 1}: "${article.title || 'Untitled'}"\n${content}`;
        })
        .join('\n\n');
      
      this.logWithTime('Generating answer using OpenAI...');
   
      const completion = await this.openai.chat.completions.create({
         model: "gpt-4o-mini",
       //  model: "grok-3-mini-beta",
        messages: [
          {
            role: "system",
            content: `You are a helpful assistant representing our company. When answering questions:
            1. Always refer to articles as 'our articles' or 'our information'
            2. Use a professional, helpful tone`
          },
          {
            role: "user",
            content: `Answer this question using only the information in the articles provided:
             "${query}"\n\nHere are the relevant articles:\n\n${context}`
          }
        ],
        temperature: 0.3,
        max_tokens: 150
      });
      console.log("answer: ", completion.choices[0].message);
      const answer = completion.choices[0].message.content.trim();
      console.log(answer)
      this.logWithTime('Answer generated successfully');
      return answer;
      
    } catch (error) {
      this.logWithTime(`Error generating answer with API: ${error.message}`, 'error');
      // Fallback to simpler response if API fails
      return 'Based on our information, please contact support for assistance with your query.';
    }
  }

  // Search for top-k relevant articles and generate an answer
  async search(query, topK = 3) {
    try {
      this.logWithTime(`Searching for articles relevant to query: "${query}"`);
      
      // Validate the query first
      if (!query || typeof query !== 'string' || query.trim() === '') {
        this.logWithTime('Empty or invalid query received', 'error');
        return {
          articles: [],
          generatedAnswer: 'Please provide a valid query to search for relevant information.'
        };
      }
      
      // Check if embeddings exist
      if (!this.embeddings || this.embeddings.length === 0) {
        this.logWithTime('No article embeddings available to search against', 'error');
        return {
          articles: [],
          generatedAnswer: 'No articles available to search through. Please ensure articles are loaded correctly.'
        };
      }
      
      console.log("query: ", query);
  
      // Extract the category filter from the query
      const categoryFilter = query.split("-")[1]?.trim();
      if (!categoryFilter) {
        this.logWithTime('No valid category filter found in query', 'error');
        return {
          articles: [],
          generatedAnswer: 'Please provide a valid category in the query to filter relevant information.'
        };
      }
  
      // Filter articles by category
      const filteredEmbeddings = this.embeddings.filter(({ article }) => 
        article.cats && article.cats.some(cat => 
          cat.toLowerCase().includes(categoryFilter.toLowerCase())
        )
      );
  
      if (filteredEmbeddings.length === 0) {
        this.logWithTime(`No articles found matching category: "${categoryFilter}"`, 'error');
        return {
          articles: [],
          generatedAnswer: `No articles found in the category "${categoryFilter}". Please try a different query.`
        };
      }
  
      // Generate embedding for the query
      const queryEmbedding = await this.pipeline(query, { 
        pooling: 'mean', 
        normalize: true,
        progress_callback: null // Disable progress logs
      });
  
      // Calculate similarity scores for filtered embeddings
      const scores = filteredEmbeddings.map(({ embedding, article }) => ({
        article,
        score: this.cosineSimilarity(queryEmbedding.data, embedding)
      }));
  
      // Find the most relevant articles
      const relevantArticles = scores
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(({ article, score }) => ({
          title: article.title || 'Untitled',
          content: article.body || '',
          did: article.did,
          sitename: article.sitename,
          cats: article.cats,
          score
        }));
  
      this.logWithTime(`Found ${relevantArticles.length} relevant articles`);
  
      // Generate an answer based on the relevant articles (AI)
      const answer = await this.generateAnswer(relevantArticles, query);
  
      // Return both relevant articles and the generated answer
      return {
        articles: relevantArticles,
        generatedAnswer: answer
      };
    } catch (error) {
      this.logWithTime(`Error during search: ${error.message}`, 'error');
      
      // Provide a more graceful failure mode, returning the error in the response
      return {
        articles: [],
        generatedAnswer: 'An error occurred while processing your query. Please try again later.',
        error: error.message
      };
    }
  }
}

export default EmbeddingManager;