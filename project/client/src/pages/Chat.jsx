// client/src/components/chatbot/ChatWindow.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, TextField, Button } from '@mui/material';
import { motion } from 'framer-motion';
import { sendMessage } from '../services/chatService';

const ChatWindow = () => {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: 'Hello! How can I help you with your wine selection today?' }
  ]);
  const [input, setInput] = useState('');
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  // Scroll to bottom whenever messages update
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Retrieve token from localStorage
    const token = localStorage.getItem('token');
    if (!token) {
      setError('You must be logged in to chat.');
      return;
    }

    // Append user's message to chat history
    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    try {
      // Call the chat API
      const data = await sendMessage(input, token);
      const assistantMessage = { sender: 'assistant', text: data.response };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending chat message:', err);
      setError('Failed to get a response from the assistant.');
    }
  };

  return (
    <Paper sx={{ p: 2, height: '70vh', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ flex: 1, overflowY: 'auto', mb: 2 }}>
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Box
              sx={{
                display: 'flex',
                justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                mb: 1
              }}
            >
              <Box
                sx={{
                  bgcolor: msg.sender === 'user' ? '#1976d2' : '#e0e0e0',
                  color: msg.sender === 'user' ? '#fff' : '#000',
                  p: 1,
                  borderRadius: 2,
                  maxWidth: '70%'
                }}
              >
                <Typography variant="body1">{msg.text}</Typography>
              </Box>
            </Box>
          </motion.div>
        ))}
        <div ref={messagesEndRef} />
      </Box>
      {error && (
        <Typography variant="body2" color="error" sx={{ mb: 1 }}>
          {error}
        </Typography>
      )}
      <Box sx={{ display: 'flex' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSend();
            }
          }}
        />
        <Button variant="contained" sx={{ ml: 1 }} onClick={handleSend}>
          Send
        </Button>
      </Box>
    </Paper>
  );
};

export default ChatWindow;
