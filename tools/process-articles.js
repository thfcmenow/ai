import { promises as fs } from 'fs';
import path from 'path';

async function processArticles() {
  try {
    const articlesDir = './articles';
    
    // Check if the articles directory exists
    try {
      await fs.access(articlesDir);
    } catch (error) {
      console.error(`Error: The directory ${articlesDir} does not exist.`);
      return;
    }
    
    // Get all files in the articles directory
    const files = await fs.readdir(articlesDir);
    
    // Filter for JSON files
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    if (jsonFiles.length === 0) {
      console.log('No JSON files found in the articles directory.');
      return;
    }
    
    console.log(`Found ${jsonFiles.length} JSON files to process.`);
    
    // Process each JSON file
    for (const file of jsonFiles) {
      const filePath = path.join(articlesDir, file);
      
      try {
        // Read the file content
        const fileContent = await fs.readFile(filePath, 'utf8');
        
        // Parse the JSON content
        let jsonData;
        try {
          jsonData = JSON.parse(fileContent);
        } catch (parseError) {
          console.error(`Error parsing JSON in file ${file}:`, parseError.message);
          continue;
        }
        
        // Check if the body field exists and is a string
        if (!jsonData.body || typeof jsonData.body !== 'string') {
          console.warn(`Warning: File ${file} does not have a valid 'body' field.`);
          continue;
        }
        
        // Decode the base64 content
        try {
          const decodedBody = Buffer.from(jsonData.body, 'base64').toString('utf8');
          
          // Escape double quotes in the HTML
          const escapedBody = decodedBody.replace(/"/g, '\\"');
          
          // Update the body field
          jsonData.body = escapedBody;
          
          // Write the updated JSON back to the file
          await fs.writeFile(filePath, JSON.stringify(jsonData, null, 2), 'utf8');
          
          console.log(`Successfully processed ${file}`);
        } catch (decodeError) {
          console.error(`Error decoding base64 in file ${file}:`, decodeError.message);
        }
      } catch (fileError) {
        console.error(`Error processing file ${file}:`, fileError.message);
      }
    }
    
    console.log('All files processed successfully.');
  } catch (error) {
    console.error('An unexpected error occurred:', error.message);
  }
}

// Execute the main function
processArticles().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

