export interface Resource {
  title: string;
  url: string;
  type: 'video' | 'article' | 'repo' | 'colab' | 'paper' | 'doc' | 'tutorial' | 'tool' | 'docs';
  author: string;
  duration?: string;
}

export interface CodeSnippet {
  language: string;
  title: string;
  code: string;
}

export interface Lesson {
  id: string;
  title: string;
  description: string;
  objectives: string[];
  resources: Resource[];
  concepts: string[];
  codeSnippets: CodeSnippet[];
  exercise: {
    description: string;
    expectedOutput: string;
  };
  difficulty: string; // Changed from strict union to string to support "Beginner-Intermediate" etc.
  durationMinutes: number;
}

export interface Module {
  id: string;
  title: string;
  description: string;
  lessons: Lesson[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}