'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  id: string
  text: string
  sender: 'user' | 'bot'
  timestamp: Date
  confidence?: number
  category?: string
  intent?: string
  sources?: Array<{
    title: string
    url: string
    score: number
    page_type: string
    semantic_topic: string
  }>
}

interface ChatResponse {
  response: string
  confidence: number
  category: string
  sources: Array<{
    title: string
    url: string
    score: number
    page_type: string
    semantic_topic: string
  }>
  timestamp: string
  query: string
  intent?: string
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      text: 'Zdravo! 👋 Ja sam MPR Chatbot - vaš asistent za pravne i administrativne informacije Ministarstva Pravde BiH. Mogu vam pomoći da pronađete zakone, pravilnike, obrasce i procedure. O čemu trebam znati?',
      sender: 'bot',
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  // Fetch suggested questions on mount
  useEffect(() => {
    const fetchSuggestedQuestions = async () => {
      try {
        const response = await fetch(`${API_URL}/suggested-questions`)
        const data = await response.json()
        setSuggestedQuestions(data)
      } catch (error) {
        console.error('Failed to fetch suggested questions:', error)
      }
    }

    fetchSuggestedQuestions()
  }, [API_URL])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (messageText: string) => {
    if (!messageText.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      sender: 'user',
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: messageText,
          language: 'bs',
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data: ChatResponse = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        sender: 'bot',
        timestamp: new Date(),
        confidence: data.confidence,
        category: data.category,
        intent: data.intent,
        sources: data.sources,
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        text: 'Izvinite, došlo je do greške pri obradi vašeg pitanja. Molim pokušajte ponovno.',
        sender: 'bot',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleSuggestedQuestion = (question: string) => {
    sendMessage(question)
  }

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-gradient-to-b from-blue-900 to-blue-800 text-white p-4 hidden md:flex flex-col">
        <div className="mb-8">
          <h1 className="text-2xl font-bold">MPR Chatbot</h1>
          <p className="text-blue-200 text-sm mt-1">Ministarstvo Pravde BiH</p>
        </div>

        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-3 text-blue-100">Često postavljana pitanja</h3>
          <div className="space-y-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSuggestedQuestion(question)}
                className="text-left text-sm p-2 rounded hover:bg-blue-700 transition-colors text-blue-50 hover:text-white"
              >
                💬 {question}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-auto pt-4 border-t border-blue-700">
          <p className="text-xs text-blue-200">
            ℹ️ Ovaj sistem nudi informacije i ne pruža pravni savjet.
          </p>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
          <h2 className="text-xl font-bold text-gray-800 flex items-center">
            🏛️ MPR Pravni Asistent
          </h2>
          <p className="text-sm text-gray-600">Digitalno usmjeravanje i podrška za korisnike</p>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex animate-slide-up ${
                message.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-lg px-4 py-3 rounded-lg ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-gray-200 text-gray-800 rounded-bl-none border border-gray-300'
                }`}
              >
                <p className="text-sm leading-relaxed">{message.text}</p>
                <p
                  className={`text-xs mt-1 ${
                    message.sender === 'user'
                      ? 'text-blue-100'
                      : 'text-gray-500'
                  }`}
                >
                  {message.timestamp.toLocaleTimeString('bs-BA', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
                {message.confidence !== undefined && (
                  <div className="text-xs text-gray-600 mt-2 space-y-1 border-t pt-1">
                    <p>
                      <span className="font-semibold">🧠 Model pouzdanost:</span> 
                      <span className={`ml-1 font-bold ${
                        message.confidence > 0.7 ? 'text-green-600' :
                        message.confidence > 0.5 ? 'text-yellow-600' :
                        'text-orange-600'
                      }`}>
                        {(message.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="text-gray-500 text-xs ml-1">({message.intent || 'nepoznata intencija'})</span>
                    </p>
                  </div>
                )}
                
                {/* Show sources if available */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-gray-300">
                    <p className="text-xs font-semibold text-gray-700 mb-2">📎 Pronađeni izvori i njihova relevantnost:</p>
                    <div className="space-y-2">
                      {message.sources.map((source, idx) => {
                        const relevancePercent = (source.score * 100);
                        const relevanceColor = 
                          relevancePercent > 80 ? 'text-green-600' :
                          relevancePercent > 60 ? 'text-blue-600' :
                          'text-gray-600';
                        
                        return (
                          <div key={idx} className="flex flex-col">
                            <a
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
                              title={source.title}
                            >
                              {idx + 1}. 🔗 {source.title || `Izvor ${idx + 1}`}
                            </a>
                            <span className={`text-xs ${relevanceColor} font-semibold`}>
                              Relevantnost: {relevancePercent.toFixed(1)}%
                            </span>
                            <span className="text-xs text-gray-500">
                              Tip: {source.page_type} • Tema: {source.semantic_topic}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 px-4 py-3 rounded-lg rounded-bl-none border border-gray-300">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4 shadow-lg">
          <div className="flex space-x-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !isLoading) {
                  sendMessage(inputValue)
                }
              }}
              placeholder="Pitajte nešto vezano za pravne procedure..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              disabled={isLoading}
            />
            <button
              onClick={() => sendMessage(inputValue)}
              disabled={isLoading || !inputValue.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
            >
              <span>Pošalji</span>
              <span>➜</span>
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            💡 Savjet: Pokušajte sa „Kako registrovati udruženje?" ili izaberite predloženo pitanje iz lijeve kolone.
          </p>
        </div>
      </div>
    </div>
  )
}
