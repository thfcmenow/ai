
// Set environment variables to suppress ONNX Runtime warnings
// This needs to be set before any ONNX/transformers imports
process.env.ONNX_RUNTIME_LOG_LEVEL = '0'; // 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
process.env.ONNX_RUNTIME_LOG_VERBOSITY_LEVEL = '0';
process.env.ONNX_RUNTIME_LOGGING_SEVERITY = 'VERBOSE';

import express from 'express';
import EmbeddingManager from './embeddings.js';
import fsPromises from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { createRequire } from 'module';
import { config } from 'dotenv';
import cors from 'cors'; // Import cors


// Near the top of your main file
const originalWrite = process.stderr.write;
process.stderr.write = function(chunk, ...args) {
  const text = chunk.toString();
  if (text.includes('CleanUnusedInitializersAndNodeArgs') || 
      text.includes('Removing initializer') ||
      text.includes('onnxruntime')) {
    return true; // Skip this output
  }
  return originalWrite.apply(this, [chunk, ...args]);
};

const app = express();
const port = 3000;

app.use(express.json());

app.use(cors({
  origin: 'http://localhost:3001' // Allow only your frontend origin
}));

// Format log messages with timestamps
function logWithTime(message, type = 'info') {
  const timestamp = new Date().toISOString();
  const prefix = type === 'error' ? '❌ ERROR:' : '✓ INFO:';
  console[type](`[${timestamp}] ${prefix} ${message}`);
}

const embeddingManager = new EmbeddingManager();

// Initialize embeddings on server start
logWithTime('Starting server and initializing embedding manager...');
embeddingManager.init()
  .then(() => {
    logWithTime('Embedding manager successfully initialized');
  })
  .catch(error => {
    logWithTime(`Failed to initialize embedding manager: ${error.message}`, 'error');
    process.exit(1); // Exit if initialization fails
  });

// Articles endpoint
app.get('/articles', async (_, res) => {
  let articles = [];
  try {
    const articlesDir = "./articles";
    const files = await fsPromises.readdir(articlesDir);
    for (const file of files) {
      const filePath = path.join(articlesDir, file);
      const data = await fsPromises.readFile(filePath, 'utf8');
      console.log(`Reading file: ${filePath}`);
      if (!data) {
        logWithTime(`Empty file: ${file}`, 'error');
        continue; // Skip empty files
      }
      const article = JSON.parse(data);
      if (!article.title || !article.desc) {
        logWithTime(`Invalid article format in ${file}`, 'error');
        continue; // Skip invalid articles
      }
      articles.push({title: article.title, desc: article.desc, site: article.sitename});
    }
    res.json(articles);
  } catch (error) {
    logWithTime(`Error reading articles: ${error.message}`, 'error');
    res.status(500).json({ error: 'Internal server error', message: error.message });
  }
  logWithTime(`Fetched ${articles.length} articles`);
})

app.get('/', (req, res) => {
  res.send('Hello World! This is the backend server for the embedding manager.');
}
);

// Query endpoint
app.get('/query/:query', async (req, res) => {
  const query = req.params.query;
  if (!query) {
    return res.status(400).json({ error: 'Query is required' });
  }

  try {
    logWithTime(`Processing query: "${query}"`);
    const results = await embeddingManager.search(query);
    logWithTime(`Query processed successfully, found ${results.articles.length} relevant articles`);
    res.json({ results });
  } catch (error) {
    logWithTime(`Error processing query "${query}": ${error.message}`, 'error');
    res.status(500).json({ error: 'Internal server error', message: error.message });
  }
});

const PORT = process.env.PORT || 3001;


app.listen(PORT, () => {
  logWithTime(`Server running at http://localhost:${PORT}`);
  logWithTime('Ready to process embedding queries');
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  logWithTime(`Uncaught exception: ${error.message}`, 'error');
  console.error(error.stack);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  logWithTime(`Unhandled promise rejection: ${reason}`, 'error');
});
