import { GoogleGenAI } from "@google/genai";
import { Lesson } from '../types';

let aiClient: GoogleGenAI | null = null;

// Initialize client with key from environment
try {
  if (process.env.API_KEY) {
    aiClient = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }
} catch (e) {
  console.error("Error initializing Gemini client", e);
}

export const explainConcept = async (concept: string, lessonContext: Lesson): Promise<string> => {
  if (!aiClient) return "API Key not configured. Please set process.env.API_KEY.";

  try {
    const prompt = `
      You are an expert AI Tutor for a course on RAG and LLM Fine-Tuning.
      The student is currently working on the lesson: "${lessonContext.title}".
      
      They have asked for an explanation of the concept: "${concept}".
      
      Provide a clear, technical, but accessible explanation (approx 150 words). 
      If applicable, use a small code analogy in Python.
    `;

    const response = await aiClient.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });
    
    return response.text || "No response generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "I encountered an error trying to explain that concept. Please check your API key.";
  }
};

export const askTutor = async (question: string, lessonContext: Lesson): Promise<string> => {
  if (!aiClient) return "API Key not configured.";

  try {
     const prompt = `
      Context: Course "RAG & Fine-Tuning Mastery". Lesson: "${lessonContext.title}".
      Objectives: ${lessonContext.objectives.join(', ')}.
      
      Student Question: "${question}"
      
      Answer as a senior ML engineer. Be practical and encourage hands-on testing.
    `;

    const response = await aiClient.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "No response.";
  } catch (error) {
    return "Error contacting the AI Tutor.";
  }
}
