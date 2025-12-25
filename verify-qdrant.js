import { QdrantClient } from '@qdrant/qdrant-js';

// Load environment variables
const qdrantUrl = "https://7dfcd22f-c548-40b3-b8a8-ced09a62d872.us-east4-0.gcp.cloud.qdrant.io";
const qdrantApiKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.x6gnR3YxLTrQv6oB60DNKkrryp7LppQRZPkOczlm2eI";

// Connect to Qdrant
const client = new QdrantClient({
  url: qdrantUrl,
  apiKey: qdrantApiKey,
});

async function verifyQdrantEmbeddings() {
  try {
    console.log("Connecting to Qdrant...\n");

    // List all collections
    const collections = await client.getCollections();
    console.log("=== Available Collections ===");
    for (const collection of collections.collections) {
      console.log(`- ${collection.name}`);
    }

    // Look for collections that might contain book embeddings
    const collectionNames = collections.collections.map(col => col.name);
    console.log("\nCollection Names:", collectionNames);

    for (const collectionName of collectionNames) {
      console.log(`\n=== Checking Collection: ${collectionName} ===`);

      try {
        // Get collection info
        const collectionInfo = await client.getCollection(collectionName);
        console.log("Status:", collectionInfo.status);
        console.log("Optimizer Status:", collectionInfo.optimizer_status);
        console.log("Indexed Vectors Count:", collectionInfo.indexed_vectors_count);
        console.log("Points Count:", collectionInfo.points_count);
        console.log("Segments Count:", collectionInfo.segments_count);

        // Check vector configuration
        console.log("Vector Config:", collectionInfo.config.params.vectors);

        // Get point count using correct method
        const countResult = await client.count(collectionName, {});
        console.log("Actual Point Count:", countResult.count);

        // Sample a few points to check structure if any exist
        if (countResult.count > 0) {
          const samplePoints = await client.scroll(collectionName, {
            limit: 3,
            with_payload: true,
            with_vector: false
          });

          console.log("Sample Points:");
          for (const point of samplePoints.points) {
            console.log("- ID:", point.id);
            console.log("  Payload keys:", Object.keys(point.payload || {}));
            console.log("  Payload example:", JSON.stringify(point.payload, null, 2));
          }

          // Run quality checks on embeddings if points exist
          console.log("\n🔍 Running quality checks...");

          // Check for zero vectors and duplicates
          const allPoints = await client.scroll(collectionName, {
            limit: Math.min(10, countResult.count), // Limit to 10 points maximum
            with_payload: false,
            with_vector: true
          });

          let zeroVectors = 0;
          let duplicateCount = 0;
          const vectorHashes = new Set();

          for (const point of allPoints.points) {
            if (point.vector) {
              // Check for zero vectors
              const isZeroVector = point.vector.every(val => val === 0);
              if (isZeroVector) zeroVectors++;

              // Check for duplicates using a hash of the vector
              const vectorHash = point.vector.join(',');
              if (vectorHashes.has(vectorHash)) {
                duplicateCount++;
              }
              vectorHashes.add(vectorHash);
            }
          }

          console.log(`Zero vectors detected: ${zeroVectors}`);
          console.log(`Duplicate vectors detected: ${duplicateCount}`);
        } else {
          console.log("⚠️  No points found in this collection - embeddings may not be ingested yet");
        }

        // Test search functionality if there are points
        if (countResult.count > 0) {
          console.log("\n🧪 Testing semantic search functionality...");

          try {
            // Get a sample point to use for search
            const samplePoints = await client.scroll(collectionName, {
              limit: 1,
              with_payload: true,
              with_vector: true
            });

            if (samplePoints.points.length > 0) {
              const samplePoint = samplePoints.points[0];
              console.log(`Testing search with vector dimension: ${Array.isArray(samplePoint.vector) ? samplePoint.vector.length : 'unknown'}`);

              // Perform a search to test functionality
              const searchResults = await client.search(collectionName, {
                vector: Array.isArray(samplePoint.vector) ? samplePoint.vector : Object.values(samplePoint.vector)[0], // Handle both named and unnamed vector formats
                limit: 3,
                with_payload: true
              });

              console.log("Search Results (top 3):");
              for (const result of searchResults) {
                console.log(`- Score: ${result.score?.toFixed(4) || 'N/A'}, ID: ${result.id}`);
                if (result.payload) {
                  const contentPreview = result.payload.text || result.payload.content || result.payload.heading || result.payload.module || 'No content found';
                  console.log(`  Content preview: ${contentPreview.substring(0, 100)}...`);
                }
              }
            }
          } catch (searchErr) {
            console.log(`⚠️  Search test failed: ${searchErr.message}`);
          }
        } else {
          // If no points exist, suggest ingestion might be needed
          console.log("💡 This collection appears to be empty - you may need to run the embedding ingestion process");
        }
      } catch (err) {
        console.log(`❌ Error checking collection ${collectionName}:`, err.message);
      }
    }

    console.log("\n✅ Qdrant verification completed!");
  } catch (error) {
    console.error("❌ Qdrant connection error:", error.message);
    console.error("Stack trace:", error.stack);
  }
}

// Run verification
verifyQdrantEmbeddings();