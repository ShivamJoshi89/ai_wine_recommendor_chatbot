// client/src/pages/Chat.jsx
import React, { useState } from 'react';
import { Container, Typography, Box, Button, Paper } from '@mui/material';
import ChatWindow from '../components/chatbot/ChatWindow';

const Chat = () => {
  const quickReplies = [
    "Show me red wines",
    "I like sparkling wine",
    "Recommend something under $30",
    "Tell me about Chardonnay"
  ];

  const [quickReplyMessage, setQuickReplyMessage] = useState('');

  const handleQuickReply = (reply) => {
    setQuickReplyMessage(reply);
    console.log("Quick reply selected:", reply);
  };

  return (
    <Container sx={{ py: 4 }}>
      <Typography variant="h4" sx={{ mb: 2, textTransform: 'uppercase', color: 'text.primary' }}>
        Chat with Our Wine Advisor
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
        {quickReplies.map((reply, idx) => (
          <Button
            key={idx}
            variant="outlined"
            onClick={() => handleQuickReply(reply)}
            sx={{
              textTransform: 'none',
              color: 'text.primary',
              borderColor: 'text.primary'
            }}
          >
            {reply}
          </Button>
        ))}
      </Box>
      
      <Paper sx={{ p: 2, backgroundColor: 'transparent', boxShadow: 'none' }}>
        {/* Render the chat window with the quick reply prop */}
        <ChatWindow quickReply={quickReplyMessage} />
      </Paper>
    </Container>
  );
};

export default Chat;
